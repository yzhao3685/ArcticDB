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

TEST(PythonHandlerData, ReadScopeKeepsNanAliveUntilWorkersFinish) {
    using namespace arcticdb;
    ScopedGILLock gil;
    py::object probe;
    std::future<Py_ssize_t> worker;
    {
        PythonReadHandlerData read_data{OutputFormat::PANDAS};
        auto& handler_data = std::any_cast<PythonHandlerData&>(read_data.get());
        probe = py::reinterpret_borrow<py::object>(handler_data.non_owning_nan_handle());
        worker = std::async(std::launch::async, [data = read_data.take_shared_data()]() mutable {
            ScopedGILLock worker_gil;
            const auto& handler = std::any_cast<const PythonHandlerData&>(*data);
            const auto count = Py_REFCNT(handler.non_owning_nan_handle());
            data.reset();
            return count;
        });
    }

    Py_ssize_t worker_count;
    {
        py::gil_scoped_release release;
        worker_count = worker.get();
    }
    EXPECT_EQ(worker_count, 2);
    EXPECT_EQ(Py_REFCNT(probe.ptr()), 1);
}
