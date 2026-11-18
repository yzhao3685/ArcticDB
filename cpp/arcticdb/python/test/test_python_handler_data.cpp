/* Copyright 2026 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software
 * will be governed by the Apache License, version 2.0.
 */

#include <arcticdb/python/gil_lock.hpp>
#include <arcticdb/python/python_handler_data.hpp>
#include <gtest/gtest.h>

#include <any>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>

TEST(PythonHandlerData, GetMissingValue) {
    using namespace arcticdb;

    ScopedGILLock gil;
    PythonReadHandlerData read_data{OutputFormat::PANDAS};
    auto& handler_data = std::any_cast<PythonHandlerData&>(read_data.get());
    EXPECT_EQ(handler_data.get_missing_value(PandasMissingValue::None), Py_None);
    EXPECT_EQ(handler_data.get_missing_value(PandasMissingValue::NaN), handler_data.non_owning_nan_handle());
}

TEST(PythonHandlerData, DestructionDoesNotWaitForGil) {
    using namespace arcticdb;
    ScopedGILLock gil;
    PythonReadHandlerData read_data{OutputFormat::PANDAS};
    auto handler_data = read_data.take_shared_data();
    // Workers drop the last copy while this thread holds the GIL, e.g. while the interpreter is finalizing
    auto destroyed = std::async(std::launch::async, [&handler_data] { handler_data.reset(); });
    const bool destroyed_without_waiting = destroyed.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    {
        // Unblock the worker if it is waiting for the GIL
        py::gil_scoped_release release;
        destroyed.wait();
    }
    EXPECT_TRUE(destroyed_without_waiting);
}

TEST(PythonHandlerData, ReleasesUnusedNan) {
    using namespace arcticdb;
    ScopedGILLock gil;
    for (const bool complete_read : {false, true}) {
        py::object probe;
        {
            PythonReadHandlerData read_data{OutputFormat::PANDAS};
            auto& handler_data = std::any_cast<PythonHandlerData&>(read_data.get());
            probe = py::reinterpret_borrow<py::object>(handler_data.non_owning_nan_handle());
            if (complete_read) {
                handler_data.apply_nan_refcount();
            }
        }
        EXPECT_EQ(Py_REFCNT(probe.ptr()), 1);
    }
}

TEST(PythonHandlerData, ReadScopeReleasesOnlyItsOwnNanReference) {
    using namespace arcticdb;
    ScopedGILLock gil;
    py::object first_slot;
    py::object second_slot;
    {
        PythonReadHandlerData read_data{OutputFormat::PANDAS};
        auto& handler_data = std::any_cast<PythonHandlerData&>(read_data.get());
        auto* nan = handler_data.non_owning_nan_handle();
        handler_data.increment_missing_refcount(PandasMissingValue::NaN, 2);
        handler_data.apply_nan_refcount();
        ASSERT_EQ(Py_REFCNT(nan), 3);
        first_slot = py::reinterpret_steal<py::object>(nan);
        second_slot = py::reinterpret_steal<py::object>(nan);
    }
    EXPECT_EQ(Py_REFCNT(first_slot.ptr()), 2);
}

namespace {

constexpr auto deferred_nans_key = "arcticdb.deferred_pandas_nans";

enum class NanCleanup : uint8_t { NextRead, InterpreterTeardown, InterpreterTeardownWithWorker };

class DeferredNanCleanupTest : public testing::TestWithParam<NanCleanup> {};

TEST_P(DeferredNanCleanupTest, ReadScopeReturnsBeforeWorkerAndReleasesNanLater) {
    using namespace arcticdb;
    ScopedGILLock gil;
    py::object probe;
    std::promise<void> allow_worker_to_finish;
    std::future<bool> worker;
    {
        PythonReadHandlerData read_data{OutputFormat::PANDAS};
        auto& handler_data = std::any_cast<PythonHandlerData&>(read_data.get());
        probe = py::reinterpret_borrow<py::object>(handler_data.non_owning_nan_handle());
        worker = std::async(
                std::launch::async,
                [data = read_data.take_shared_data(), ready = allow_worker_to_finish.get_future()]() mutable {
                    const bool read_returned = ready.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
                    data.reset();
                    return read_returned;
                }
        );
    }

    EXPECT_EQ(Py_REFCNT(probe.ptr()), 2);
    auto* state = PyInterpreterState_GetDict(PyInterpreterState_Get());
    const auto cleanup = GetParam();
    if (cleanup == NanCleanup::InterpreterTeardownWithWorker) {
        EXPECT_EQ(PyDict_DelItemString(state, deferred_nans_key), 0);
        EXPECT_EQ(Py_REFCNT(probe.ptr()), 2);
    }
    allow_worker_to_finish.set_value();
    EXPECT_TRUE(worker.get());
    EXPECT_EQ(Py_REFCNT(probe.ptr()), 2);

    if (cleanup == NanCleanup::NextRead) {
        PythonReadHandlerData next_read{OutputFormat::PANDAS};
    } else if (cleanup == NanCleanup::InterpreterTeardown) {
        // Release the capsule just as interpreter teardown does.
        ASSERT_EQ(PyDict_DelItemString(state, deferred_nans_key), 0);
    } else {
        // This test is still in a live interpreter after simulating teardown.
        ASSERT_EQ(Py_REFCNT(probe.ptr()), 2);
        Py_DECREF(probe.ptr());
    }
    EXPECT_EQ(Py_REFCNT(probe.ptr()), 1);
}

INSTANTIATE_TEST_SUITE_P(
        PythonHandlerData, DeferredNanCleanupTest,
        testing::Values(
                NanCleanup::NextRead, NanCleanup::InterpreterTeardown, NanCleanup::InterpreterTeardownWithWorker
        )
);

} // namespace

TEST(PythonHandlerDataDeathTest, DeferredNanCleanupUsesTheCurrentInterpreter) {
    using namespace arcticdb;
    // Creating a subinterpreter changes PyGILState_Check behavior in the process.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_EXIT(
            {
                ScopedGILLock gil;
                PythonReadHandlerData main_read{OutputFormat::PANDAS};
                auto* main_thread = PyThreadState_Get();
                auto* main_state = PyInterpreterState_GetDict(PyInterpreterState_Get());
                auto* main_queue = PyDict_GetItemString(main_state, deferred_nans_key);

                auto* subinterpreter = Py_NewInterpreter();
                if (!subinterpreter || !main_queue) {
                    std::_Exit(EXIT_FAILURE);
                }
                bool isolated;
                std::shared_ptr<std::any> child_borrower;
                {
                    PythonReadHandlerData child_read{OutputFormat::PANDAS};
                    auto* child_state = PyInterpreterState_GetDict(PyInterpreterState_Get());
                    auto* child_queue = PyDict_GetItemString(child_state, deferred_nans_key);
                    isolated = child_queue && child_queue != main_queue;
                    child_borrower = child_read.take_shared_data();
                }
                Py_EndInterpreter(subinterpreter);
                PyThreadState_Swap(main_thread);
                child_borrower.reset();

                const bool main_queue_preserved = PyDict_GetItemString(main_state, deferred_nans_key) == main_queue;
                std::_Exit(isolated && main_queue_preserved ? EXIT_SUCCESS : EXIT_FAILURE);
            },
            testing::ExitedWithCode(EXIT_SUCCESS),
            ""
    );
}
