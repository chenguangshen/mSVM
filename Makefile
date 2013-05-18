CXX = gcc
#CFLAGS = -g -Wall -Wconversion -O3 -fPIC
CFLAG1 = -Wall -Wconversion -static -Os -O2 -fno-inline -fomit-frame-pointer -fno-zero-initialized-in-bss  -fmerge-constants # -fPIC
PACK = -fpack-struct

test_svm:
	$(CXX) $(CFLAG1) $(PACK) svm_simp.c -o svm -lm 

test_svm_cpp:
	$(CXX) $(CFLAG1) $(PACK) svm_simp.cpp -o svm -lm 

test_svm_short:
	$(CXX) $(CFLAG1) $(PACK) svm_simp_short.c -o svm -lm 

test_predict: 
	$(CXX) $(CFLAGS) -c svm.cpp -o svm.o
	$(CXX) $(CFLAGS) test_svm_predict.c svm.o -o test_svm_predict -lm

select_sv:
	$(CXX) $(CFLAGS) sv_selector.cpp -o sv_selector -lm

test_svm_comp:
	$(CXX) $(CFLAGS) svm_comp.cpp -o svm_comp -lm

test_svm_dim:
	$(CXX) $(CFLAGS) svm_dim.cpp -o svm_dim -lm

test_svm_llvm:
	hexagon-clang -mv5 $(CFLAG1) -g svm_simp.c -o svm_simp_llvm -lm -lhexagon

test_svm_hexagon:
	hexagon-gcc -mv5 $(CFLAG1) -g -Wall -Wconversion svm_simp.c -o svm_simp_hexagon -lm -lhexagon

test_svm_hexagon_cpp:
	hexagon-g++ -mv5 $(CFLAG1) $(PACK) -Wall -Wconversion svm_simp.cpp -o svm_simp_hexagon -lm -lhexagon

test_svm_hexagon_short:
	hexagon-gcc -mv5 $(CFLAG1) svm_simp_short.c -o svm_hexagon_short -lm -lhexagon

test_svm_llvm_short:
	hexagon-clang -mv5 $(CFLAG1) svm_simp_short.c -o svm_hexagon_short_llvm -lm -lhexagon

sim:
	hexagon-sim -mv5 --timing --qprof qprof_description svm_simp_hexagon

qprof:
	hexagon-profiler-gui --qproffile qprof001.out --symfile svm_simp_hexagon

clean:
	rm -f *~ *.o svm_comp svm_simp_hexagon svm_dim sv_selector svm_hexagon_short_llvm svm_hexagon_short svm-train svm_simp_llvm svm-predict svm svm-scale test_svm_simp_predict test_svm_predict
