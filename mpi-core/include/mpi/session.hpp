#pragma once

#include <mpi.h>

#if defined(MPI_VERSION) && MPI_VERSION >= 4

#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "mpi/comm.hpp"
#include "mpi/error.hpp"
#include "mpi/group.hpp"

namespace mpi::experimental {

// ── Forward declarations ──────────────────────────────────────────────────────

class session_view;
class session;

// ── pset_range ────────────────────────────────────────────────────────────────

/// @brief Lazy input range over the process set names visible to a session.
///
/// Calls `MPI_Session_get_num_psets` once on `begin()`, then fetches each name
/// on demand via `MPI_Session_get_nth_pset`. No names are stored eagerly.
///
/// The session and info object must outlive the range and its iterators.
class pset_range {
public:
    /// @brief Input iterator over process set names.
    class iterator {
    public:
        using value_type       = std::string;
        using difference_type  = std::ptrdiff_t;
        using iterator_concept = std::input_iterator_tag;

        [[nodiscard]] std::string const& operator*() const noexcept { return _current; }
        [[nodiscard]] std::string const* operator->() const noexcept { return &_current; }

        iterator& operator++() {
            ++_index;
            if (_index < _total) {
                _load_current();
            }
            return *this;
        }

        void operator++(int) { ++(*this); }

        [[nodiscard]] bool operator==(std::default_sentinel_t) const noexcept {
            return _index >= _total;
        }

        [[nodiscard]] friend bool operator==(std::default_sentinel_t s, iterator const& it) noexcept {
            return it == s;
        }

    private:
        MPI_Session _session = MPI_SESSION_NULL;
        MPI_Info    _info    = MPI_INFO_NULL;
        int         _total   = 0;
        int         _index   = 0;
        std::string _current;

        /// @brief Fetch the name at `_index` into `_current`.
        ///
        /// Uses the two-phase pattern: first call with null pset_name to obtain
        /// the required length, then a second call to fill the string buffer.
        /// MPI may or may not include the null terminator in the reported length;
        /// we trim any trailing '\0' characters defensively.
        void _load_current() {
            int len = 0;
            int err = MPI_Session_get_nth_pset(_session, _info, _index, &len, nullptr);
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
            // Allocate len+1 bytes: MPI may or may not include the null terminator
            // in the reported length; the extra byte ensures we never overflow.
            _current.assign(static_cast<std::size_t>(len) + 1, '\0');
            int buf_len = len + 1;
            err         = MPI_Session_get_nth_pset(_session, _info, _index, &buf_len, _current.data());
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
            while (!_current.empty() && _current.back() == '\0') {
                _current.pop_back();
            }
        }

        friend class pset_range;
    };

    static_assert(std::input_iterator<iterator>);

    /// @brief Construct a range over the psets visible to `session`.
    ///
    /// @param session The session whose psets to enumerate (must outlive the range).
    /// @param info    MPI info hints passed to `MPI_Session_get_num_psets` /
    ///                `MPI_Session_get_nth_pset` (defaults to `MPI_INFO_NULL`).
    explicit pset_range(MPI_Session session, MPI_Info info = MPI_INFO_NULL) noexcept
        : _session(session), _info(info) {}

    /// @return An input iterator positioned at the first process set name.
    ///
    /// Calls `MPI_Session_get_num_psets` to determine the total count, then
    /// fetches the first name. Subsequent increments each make one pair of
    /// `MPI_Session_get_nth_pset` calls.
    /// @throws mpi_error if `MPI_Session_get_num_psets` fails.
    [[nodiscard]] iterator begin() const {
        iterator it;
        it._session = _session;
        it._info    = _info;
        int err     = MPI_Session_get_num_psets(_session, _info, &it._total);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        it._index = 0;
        if (it._total > 0) {
            it._load_current();
        }
        return it;
    }

    /// @return The sentinel marking the end of the range.
    [[nodiscard]] std::default_sentinel_t end() const noexcept { return std::default_sentinel; }

private:
    MPI_Session _session;
    MPI_Info    _info;
};

// ── session_accessors ─────────────────────────────────────────────────────────

/// @brief CRTP mixin providing read-only accessors for any session wrapper.
///
/// @tparam Derived Must implement `MPI_Session mpi_handle() const noexcept`.
template <typename Derived>
class session_accessors {
    [[nodiscard]] MPI_Session sess() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @brief Create an owned group from a named process set.
    ///
    /// Calls `MPI_Group_from_session_pset`. Returns `std::nullopt` if the calling
    /// process is not a member of the named process set — MPI sets the output
    /// group to `MPI_GROUP_EMPTY` in that case. Returns the owned group otherwise.
    ///
    /// This is a local (non-collective) operation.
    ///
    /// @param pset_name Name of the process set (e.g., `"mpi://WORLD"`).
    /// @return The owned group, or `std::nullopt` if this process is not a member.
    /// @throws mpi_error if the MPI call fails (e.g., unknown pset name).
    [[nodiscard]] std::optional<group> group_from_pset(std::string_view pset_name) const {
        // string_view is not guaranteed null-terminated; check before using data() directly.
        std::string tmp_storage;
        char const* cstr = (pset_name.data()[pset_name.size()] == '\0')
                               ? pset_name.data()
                               : (tmp_storage = pset_name, tmp_storage.c_str());
        MPI_Group g   = MPI_GROUP_EMPTY;
        int       err = MPI_Group_from_session_pset(sess(), cstr, &g);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        if (g == MPI_GROUP_EMPTY) {
            return std::nullopt;
        }
        return group::from_native(g);
    }

    /// @brief Collective: create a communicator from a named process set.
    ///
    /// Convenience shortcut for:
    /// @code
    /// auto g = s.group_from_pset(pset_name);
    /// if (g) auto c = comm::from_group(*g, tag, info);
    /// @endcode
    ///
    /// Returns `std::nullopt` if the calling process is not a member of the pset.
    /// All processes that *are* members must call this collectively with the same
    /// `tag`. Processes that receive `std::nullopt` must not call this.
    ///
    /// @tparam Info Any type satisfying `convertible_to_mpi_handle<MPI_Info>`:
    ///              raw `MPI_Info`, `info`, `info_view`, or any third-party wrapper.
    /// @param pset_name Name of the process set (e.g., `"mpi://WORLD"`).
    /// @param tag       Application-defined string tag (must match across all callers).
    /// @param info      MPI info hints (defaults to `MPI_INFO_NULL`).
    /// @return The new communicator, or `std::nullopt` if this process is not a member.
    /// @throws mpi_error if either MPI call fails.
    template <convertible_to_mpi_handle<MPI_Info> Info = MPI_Info>
    [[nodiscard]] std::optional<comm> comm_from_pset(
        std::string_view pset_name,
        std::string_view tag  = "",
        Info             info = MPI_INFO_NULL
    ) const {
        auto g = group_from_pset(pset_name);
        if (!g) {
            return std::nullopt;
        }
        return comm::from_group(*g, tag, info);
    }

    /// @brief Enumerate all process set names visible to this session.
    ///
    /// Returns a lazy input range. `MPI_Session_get_num_psets` is called once
    /// when iteration begins; each increment fetches the next name on demand via
    /// `MPI_Session_get_nth_pset`. No names are loaded until iteration starts.
    ///
    /// Accepts a raw `MPI_Info` handle only — passing an owning wrapper temporary
    /// would leave the range with a dangling handle after this call returns.
    ///
    /// @param info Optional MPI info hints (defaults to `MPI_INFO_NULL`).
    [[nodiscard]] pset_range psets(MPI_Info info = MPI_INFO_NULL) const {
        return pset_range{sess(), info};
    }
};

// ── session_view ──────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Session`.
///
/// Does not finalize the session on destruction — use the owning `session` for that.
class session_view : public session_accessors<session_view> {
public:
    /// @brief Construct from a raw `MPI_Session`. The session must outlive this view.
    explicit session_view(MPI_Session s) noexcept : _session(s) {}

    /// @return The underlying `MPI_Session` (for accessor dispatch).
    [[nodiscard]] MPI_Session mpi_handle() const noexcept { return _session; }

private:
    MPI_Session _session;
};

// ── session ───────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Session` (MPI-4 sessions model).
///
/// Move-only. Calls `MPI_Session_finalize` on destruction unless the stored
/// handle is `MPI_SESSION_NULL` or MPI has already been finalized.
///
/// Unlike `kamping::v2::environment`, a session does not require `MPI_Init`
/// to have been called — the sessions model allows MPI to be initialized via
/// sessions alone. When used alongside the traditional model, the session should
/// be finalized before `MPI_Finalize`.
///
/// @par Usage
/// @code
/// namespace psets = kamping::v2::psets;
///
/// mpi::experimental::session s;
///
/// // Create a comm directly from a pset name (most concise):
/// auto c = s.comm_from_pset(psets::world);
///
/// // Or work through the group explicitly:
/// auto g = s.group_from_pset(psets::world);  // std::optional<group>
/// if (g) {
///     auto c2 = comm::from_group(*g, "my-tag");
/// }
/// @endcode
class session : public session_accessors<session> {
public:
    /// @brief Initialize a session with default info and the return-errors error handler.
    ///
    /// @throws mpi_error if `MPI_Session_init` fails.
    session() {
        int err = MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &_session);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }

    session(session const&)            = delete;
    session& operator=(session const&) = delete;

    /// @brief Move constructor — transfers ownership; moved-from holds `MPI_SESSION_NULL`.
    session(session&& o) noexcept : _session(std::exchange(o._session, MPI_SESSION_NULL)) {}

    /// @brief Move assignment — finalizes the current session then transfers ownership.
    session& operator=(session&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _session = std::exchange(o._session, MPI_SESSION_NULL);
        }
        return *this;
    }

    /// @brief Finalize the session unless already done or MPI is already finalized.
    ~session() noexcept { free_if_valid(); }

    /// @brief Adopt an already-initialized `MPI_Session` handle (takes ownership).
    ///
    /// Use when you have called `MPI_Session_init` directly and want RAII management.
    [[nodiscard]] static session from_native(MPI_Session s) noexcept {
        return session(s, adopt_t{});
    }

    /// @brief Explicitly finalize the session.
    ///
    /// After this call `*this` holds `MPI_SESSION_NULL` — the destructor becomes a
    /// no-op. Use this instead of relying on the destructor when you need to handle
    /// errors from `MPI_Session_finalize`.
    ///
    /// The handle is exchanged to `MPI_SESSION_NULL` before the MPI call so the
    /// destructor cannot double-free even if this throws.
    ///
    /// @throws mpi_error if `MPI_Session_finalize` fails.
    void finalize() {
        if (_session != MPI_SESSION_NULL) {
            MPI_Session tmp = std::exchange(_session, MPI_SESSION_NULL);
            int         err = MPI_Session_finalize(&tmp);
            if (err != MPI_SUCCESS) {
                throw mpi_error(err);
            }
        }
    }

    /// @brief Relinquish ownership; returns the raw `MPI_Session` handle.
    ///
    /// Leaves `*this` as `MPI_SESSION_NULL`. The caller is responsible for calling
    /// `MPI_Session_finalize` on the returned handle.
    [[nodiscard]] MPI_Session release() noexcept {
        return std::exchange(_session, MPI_SESSION_NULL);
    }

    /// @brief Implicit conversion to a non-owning view.
    operator session_view() const noexcept { return session_view{_session}; }

    /// @return The underlying `MPI_Session` (for accessor dispatch).
    [[nodiscard]] MPI_Session mpi_handle() const noexcept { return _session; }

private:
    struct adopt_t {};

    session(MPI_Session s, adopt_t) noexcept : _session(s) {}

    /// @brief Finalize the session if it is non-null and MPI has not yet been finalized.
    ///
    /// The `MPI_Finalized` guard prevents undefined behaviour when the session
    /// outlives `MPI_Finalize` (a programming error, but one that should not cause
    /// a crash in the destructor).
    void free_if_valid() noexcept {
        if (_session != MPI_SESSION_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Session_finalize(&_session);
            }
            _session = MPI_SESSION_NULL;
        }
    }

    MPI_Session _session = MPI_SESSION_NULL;
};

} // namespace mpi::experimental

#endif // MPI_VERSION >= 4
