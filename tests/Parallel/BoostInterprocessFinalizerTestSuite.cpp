#define BOOST_TEST_MODULE BoostInterProcessFinalizerTestSuite
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <thread>

#include "hops/Parallel/BoostInterprocessFinalizer.hpp"

void test_write_shm(
        const std::string &name,
        double value,
        int process) {
    hops::BoostInterprocessFinalizer::initializeAndQueueFinalizeAtExit();

    boost::interprocess::shared_memory_object shm_obj(
            boost::interprocess::open_only,
            hops::BoostInterprocessFinalizer::shared_memory_name,
            boost::interprocess::read_write);

    long shm_size = 2 * sizeof(double);
    shm_obj.truncate(shm_size);
    boost::interprocess::mapped_region region(
            shm_obj,
            boost::interprocess::read_write);
    *((double*) region.get_address() + process) = value;
}

void test_read_shm(const std::string &name, int process) {
    boost::interprocess::shared_memory_object shm_obj(
            boost::interprocess::open_only,
            hops::BoostInterprocessFinalizer::shared_memory_name,
            boost::interprocess::read_write);
    long shm_size = 2 * sizeof(double);
    shm_obj.truncate(shm_size);
    boost::interprocess::mapped_region region(
            shm_obj,
            boost::interprocess::read_only);
    auto value = *(static_cast<double*>(region.get_address()) + process);
    std::cout << "thread " << name << " read val " << value << std::endl;
}

BOOST_AUTO_TEST_SUITE(BoostInterprocessFinalizer)

    BOOST_AUTO_TEST_CASE(TestSharingMemory) {

        std::string name0 = "test0";
        double value0 = 0;
        std::string name1 = "test1";
        double value1 = 1;

        std::thread thread0(test_write_shm, name0, value0, 0);
        std::thread thread1(test_write_shm, name1, value1, 1);

        thread0.join();
        thread1.join();

        std::thread thread2(test_read_shm, name0, 1);
        std::thread thread3(test_read_shm, name1, 0);

        thread2.join();
        thread3.join();
    }

BOOST_AUTO_TEST_SUITE_END()
