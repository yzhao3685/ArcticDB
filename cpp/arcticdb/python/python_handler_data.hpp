#pragma once

#include <pybind11/pybind11.h>
#include <arcticdb/pipeline/output_config.hpp>
#include <arcticdb/util/preconditions.hpp>
#include <arcticdb/util/preprocess.hpp>
#include <arcticdb/util/type_handler.hpp>

#include <any>
#include <atomic>
#include <limits>
#include <memory>
#include <utility>

// Python 3.12 (PEP 683) made Py_None immortal: Py_INCREF on it is a no-op, so we no longer
// need to track how many times it was used and replay the INCREFs at end-of-read.
#define ARCTICDB_PY_NONE_IMMORTAL_HEX 0x030C0000

namespace arcticdb {

namespace py = pybind11;

inline PyObject* create_py_nan() {
    util::check(PyGILState_Check() != 0, "Expected GIL to be held when allocating Python nan");
    auto ptr = PyFloat_FromDouble(std::numeric_limits<double>::quiet_NaN());
    util::check(ptr != nullptr, "Got null nan ptr");
    return ptr;
}

struct PythonHandlerData {
    explicit PythonHandlerData(PyObject* nan) : py_nan_(nan) {}

    void increment_missing_refcount(PandasMissingValue value, size_t increment) {
        if (value == PandasMissingValue::NaN) {
            nan_refcount_->fetch_add(increment, std::memory_order_relaxed);
        } else {
#if PY_VERSION_HEX < ARCTICDB_PY_NONE_IMMORTAL_HEX
            none_refcount_->fetch_add(increment, std::memory_order_relaxed);
#else
            (void)increment;
#endif
        }
    }

    bool is_nan_initialized() const { return static_cast<bool>(py_nan_); }

    PyObject* non_owning_nan_handle() const { return py_nan_; }

    PyObject* get_missing_value(PandasMissingValue value) const {
        return value == PandasMissingValue::NaN ? non_owning_nan_handle() : Py_None;
    }

    /// The GIL must be acquired when this is called as it changes the refcount of the global static None variable which
    /// can be used by other Python threads
    void apply_none_refcount() {
#if PY_VERSION_HEX < ARCTICDB_PY_NONE_IMMORTAL_HEX
        const size_t cnt = none_refcount_->exchange(0, std::memory_order_acq_rel);
        internal::check<ErrorCode::E_ASSERTION_FAILURE>(
                PyGILState_Check(), "The thread incrementing None refcount must hold the GIL"
        );
        for (size_t i = 0; i < cnt; ++i) {
            Py_INCREF(Py_None);
        }
#endif
    }

    /// The read scope owns this NaN exclusively until the output is handed to Python.
    void apply_nan_refcount() {
        const size_t count = nan_refcount_->exchange(0, std::memory_order_acq_rel);
        if (count > 0) {
            // Include the initial reference held by the read scope.
            Py_SET_REFCNT(py_nan_, count + 1);
        }
    }

  private:
    std::shared_ptr<std::atomic<uint64_t>> none_refcount_ = std::make_shared<std::atomic<uint64_t>>(0);
    std::shared_ptr<std::atomic<uint64_t>> nan_refcount_ = std::make_shared<std::atomic<uint64_t>>(0);
    PyObject* py_nan_;
};

inline void apply_global_refcounts(std::any& handler_data, OutputFormat output_format) {
    if (output_format == OutputFormat::PANDAS) {
        PythonHandlerData& python_handler_data = std::any_cast<PythonHandlerData&>(handler_data);
        python_handler_data.apply_nan_refcount();
        python_handler_data.apply_none_refcount();
    }
}

// Keep the owning reference on the caller's GIL-held stack while worker data borrows it.
class ARCTICDB_VISIBILITY_HIDDEN PythonReadHandlerData {
  public:
    explicit PythonReadHandlerData(OutputFormat format) :
        nan_(format == OutputFormat::PANDAS ? py::reinterpret_steal<py::object>(create_py_nan()) : py::object{}),
        data_(nan_ ? std::any(std::in_place_type<PythonHandlerData>, nan_.ptr())
                   : TypeHandlerRegistry::instance()->get_handler_data(format)) {}

    ~PythonReadHandlerData() {
        // A failed read can return before sibling decode tasks have finished.
        if (workers_done_ && !workers_done_->load(std::memory_order_acquire)) {
            auto* thread_state = PyEval_SaveThread();
            workers_done_->wait(false, std::memory_order_acquire);
            PyEval_RestoreThread(thread_state);
        }
    }

    PythonReadHandlerData(const PythonReadHandlerData&) = delete;
    PythonReadHandlerData& operator=(const PythonReadHandlerData&) = delete;

    std::any& get() { return data_; }

    std::shared_ptr<std::any> take_shared_data() {
        if (!nan_) {
            return std::make_shared<std::any>(std::move(data_));
        }

        auto done = std::make_shared<std::atomic<bool>>(false);
        auto shared = std::make_shared<SharedData>(std::move(data_), done);
        workers_done_ = std::move(done);
        auto* data = &shared->data;
        return {std::move(shared), data};
    }

  private:
    struct SharedData {
        SharedData(std::any&& handler_data, const std::shared_ptr<std::atomic<bool>>& complete) :
            data(std::move(handler_data)),
            done(complete) {}

        ~SharedData() {
            done->store(true, std::memory_order_release);
            done->notify_one();
        }

        std::any data;
        std::shared_ptr<std::atomic<bool>> done;
    };

    py::object nan_;
    std::any data_;
    std::shared_ptr<std::atomic<bool>> workers_done_;
};

} // namespace arcticdb
