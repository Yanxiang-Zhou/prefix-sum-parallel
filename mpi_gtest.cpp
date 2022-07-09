
#include <mpi.h>
#include <iostream>
#include "gtest-mpi-listener/gtest-mpi-listener.hpp"

int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Add object that will finalize MPI on exit; Google Test owns this pointer
  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

  // Get the event listener list.
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener: the default printer and the default XML printer
  ::testing::TestEventListener *l = listeners.Release(listeners.default_result_printer());
    ::testing::TestEventListener *l2 = listeners.Release(listeners.default_xml_generator());


  if (l2 == NULL) {
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD, true));
  }
  else {
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD, false));
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l2, MPI_COMM_WORLD, true));
  }

  // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
  // pass and 1 if some test fails.
  int status = RUN_ALL_TESTS();
  status++;

  return 0;
}
       