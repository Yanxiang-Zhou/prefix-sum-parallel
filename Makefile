CXX=mpic++
CCFLAGS=-Wall --std=c++17 -g -O3
LDFLAGS=

GTEST_DIR = ./gtest
CCFLAGS += -I"$(GTEST_DIR)" -I"$(GTEST_DIR)/src" -I"$(GTEST_DIR)/include"


mpi_tests: mpi_tests.o mpi_gtest.o gtest-all.o collective.o utils.o
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^


test_all: mpi_tests
	mpirun -np 16 --oversubscribe ./mpi_tests

test_mat: mpi_tests
	mpirun -np 16 --oversubscribe ./mpi_tests --gtest_filter='*Mat*'

test_prefix: mpi_tests
	mpirun -np 16 --oversubscribe ./mpi_tests --gtest_filter='*PrefixSum*:*LeftNonzero*'-'*Mat*'

test_broadcast: mpi_tests
	mpirun -np 16 --oversubscribe ./mpi_tests --gtest_filter='*Bcast*'-'*Mat*'

%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

clean:
	rm -f *.o mpi_tests test_detail.json

gtest-all.o : $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/include/gtest/gtest.h
	$(CXX) $(CCFLAGS) -c $<

