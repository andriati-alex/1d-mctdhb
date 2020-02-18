
# ***************************** MAKEFILE ***************************** #

# $ make main_program



obj_linalg = memoryHandling.o 	  \
			 array_operations.o   \
			 matrix_operations.o  \
			 tridiagonal_solver.o

obj_mctdhb = $(obj_linalg)          \
			 inout.o                \
			 interpolation.o        \
			 calculus.o             \
			 linear_potential.o     \
		 	 configurationalSpace.o \
			 structureSetup.o       \
			 observables.o          \
		 	 integrators.o

linalg_header = include/dataStructures.h	 \
				include/memoryHandling.h     \
				include/array_operations.h   \
				include/matrix_operations.h  \
	  		    include/tridiagonal_solver.h

mctdhb_header = $(linalg_header) 	    	     \
				include/inout.h                  \
				include/interpolation.h          \
				include/calculus.h		         \
		 		include/linear_potential.h       \
				include/configurationalSpace.h   \
				include/structureSetup.h         \
				include/observables.h            \
				include/integrators.h





   # ------------------------------------------------------------------ #

                         ###     EXECUTABLES     ###

   # ------------------------------------------------------------------ #



MCTDHB : libmctdhb.a exe/time_evolution.c include/integrators.h
	icc -o MCTDHB exe/time_evolution.c -L${MKLROOT}/lib/intel64 \
		-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -qopenmp \
		-L./lib -I./include -lmctdhb -lm -O3





# Libraries to be linked
# ----------------------

libmctdhb.a : $(obj_mctdhb)
	ar rcs libmctdhb.a $(obj_mctdhb)
	mv libmctdhb.a lib
	mv $(obj_mctdhb) build





# Object files to the library
# ---------------------------

memoryHandling.o : src/memoryHandling.c
	icc -c -O3 -I./include src/memoryHandling.c



array_operations.o : src/array_operations.c
	icc -c -O3 -qopenmp -I./include src/array_operations.c -lm



matrix_operations.o : src/matrix_operations.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/matrix_operations.c



tridiagonal_solver.o : src/tridiagonal_solver.c
	icc -c -O3 -qopenmp -I./include src/tridiagonal_solver.c



linear_potential.o : src/linear_potential.c
	icc -c -O3 -I./include src/linear_potential.c



calculus.o : src/calculus.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/calculus.c -lm



integrators.o : src/integrators.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/integrators.c



configurationalSpace.o : src/configurationalSpace.c
	icc -c -O3 -qopenmp -I./include src/configurationalSpace.c



observables.o : src/observables.c
	icc -c -O3 -qopenmp -I./include src/observables.c -lm



structureSetup.o : src/structureSetup.c
	icc -c -O3 -I./include src/structureSetup.c



inout.o : src/inout.c
	icc -c -O3 -I./include src/inout.c



interpolation.o : src/interpolation.c
	icc -c -O3 -I./include src/interpolation.c



clean :
	-rm build/*.o
	-rm lib/lib*
	-rm MCTDHB
