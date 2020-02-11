
# ***************************** MAKEFILE ***************************** #

# $ make main_program



obj_linalg = array_memory.o 	  \
			 array_operations.o   \
			 matrix_operations.o  \
		   	 iterative_solver.o   \
			 tridiagonal_solver.o

obj_mctdhb = $(obj_linalg)             \
			 inout.o                   \
			 interpolation.o           \
			 calculus.o                \
			 linear_potential.o        \
		 	 manybody_configurations.o \
			 data_structure.o          \
			 observables.o             \
		 	 integrator_routine.o

linalg_header = include/array.h 			 \
				include/array_memory.h		 \
				include/array_operations.h   \
				include/matrix_operations.h  \
	  		    include/tridiagonal_solver.h \
				include/iterative_solver.h

mctdhb_header = $(linalg_header) 	    	      \
				include/inout.h                   \
				include/interpolation.h           \
				include/calculus.h		          \
		 		include/linear_potential.h        \
				include/manybody_configurations.h \
				include/data_structure.h          \
				include/observables.h             \
				include/integrator_routine.h





   # ------------------------------------------------------------------ #

                         ###     EXECUTABLES     ###

   # ------------------------------------------------------------------ #



MCTDHB : libmctdhb.a exe/time_evolution.c include/integrator_routine.h
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

array_memory.o : src/array_memory.c
	icc -c -O3 -I./include src/array_memory.c



array_operations.o : src/array_operations.c
	icc -c -O3 -qopenmp -I./include src/array_operations.c -lm



matrix_operations.o : src/matrix_operations.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/matrix_operations.c



tridiagonal_solver.o : src/tridiagonal_solver.c
	icc -c -O3 -qopenmp -I./include src/tridiagonal_solver.c



iterative_solver.o : src/iterative_solver.c
	icc -c -O3 -qopenmp -I./include src/iterative_solver.c



linear_potential.o : src/linear_potential.c
	icc -c -O3 -I./include src/linear_potential.c



calculus.o : src/calculus.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/calculus.c -lm



integrator_routine.o : src/integrator_routine.c
	icc -c -O3 -qopenmp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
		-I./include src/integrator_routine.c



manybody_configurations.o : src/manybody_configurations.c
	icc -c -O3 -qopenmp -I./include src/manybody_configurations.c



observables.o : src/observables.c
	icc -c -O3 -qopenmp -I./include src/observables.c -lm



data_structure.o : src/data_structure.c
	icc -c -O3 -I./include src/data_structure.c



inout.o : src/inout.c
	icc -c -O3 -I./include src/inout.c



interpolation.o : src/interpolation.c
	icc -c -O3 -I./include src/interpolation.c



clean :
	-rm build/*.o
	-rm lib/lib*
	-rm MCTDHB
