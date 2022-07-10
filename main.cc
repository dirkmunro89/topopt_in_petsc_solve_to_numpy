#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//
#include "Filter.h"
#include "LinearElasticity.h"
#include "MMA.h"
#include "MPIIO.h"
#include "TopOpt.h"
#include "mpi.h"
#include <petsc.h>
//
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include <Python.h>
#include "numpy/arrayobject.h"
//
/*
Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

Updated: June 2019, Niels Aage
Copyright (C) 2013-2019,

Disclaimer:
The authors reserves all rights but does not guaranty that the code is
free from errors. Furthermore, we shall not be liable in any event
caused by the use of the program.
 */

static char help[] = "3D TopOpt using KSP-MG on PETSc's DMDA (structured grids) \n";

int main(int argc, char* argv[]) {


////

/*
    setenv("PYTHONPATH", ".", 0);

    Py_Initialize();
    import_array();

    // Build the 2D array in C++
    const int SIZE = 3;
    npy_intp dims[2]{SIZE, SIZE};
    const int ND = 2;
    long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };

    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            c_arr[i][j] = i + j;}
    }
    // Convert it to a NumPy array.
    PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));

    // import mymodule
    const char *module_name = "mymodule";
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    // import function
    const char *func_name = "array_tutorial";
    PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
    PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
    PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);

    // Convert back to C++ array and print.
    int len = PyArray_SHAPE(np_ret)[0];
    long double* c_out;
    c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
    std::cout << "Printing output array - C++" << std::endl;
    for (int i = 0; i < len; i++){
        std::cout << c_out[i] << ' ';
    }
    std::cout << std::endl << std::endl;

    // import function without arguments
    const char *func_name2 = "myfunction";
    PyObject *pFunc2 = PyObject_GetAttrString(pModule, func_name2);
    PyObject *pReturn2 = PyObject_CallFunctionObjArgs(pFunc2, NULL);
    PyArrayObject *np_ret2 = reinterpret_cast<PyArrayObject*>(pReturn2);

    // convert back to C++ array and print
    int len2 = PyArray_SHAPE(np_ret2)[0];
    long double* c_out2;
    c_out2 = reinterpret_cast<long double*>(PyArray_DATA(np_ret2));
    std::cout << "Printing output array 2 - C++" << std::endl;
    for (int i = 0; i < len2; i++){
        std::cout << c_out2[i] << ' ';
    }
    std::cout << std::endl << std::endl;

    Py_Finalize();
*/
////

    double      t1, t2;
    // Error code for debugging
    PetscErrorCode ierr;

    // Initialize PETSc / MPI and pass input arguments to PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, help);

    // STEP 1: THE OPTIMIZATION PARAMETERS, DATA AND MESH (!!! THE DMDA !!!)
    t1 = MPI_Wtime();
    TopOpt* sim = new TopOpt();
    t2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"TopOpt %f\n",t2-t1);

    // STEP 2: THE PHYSICS
    t1 = MPI_Wtime();
    LinearElasticity* physics = new LinearElasticity(sim->da_nodes);
    t2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"LinElas %f\n",t2-t1);

    // STEP 3: THE FILTERING
    t1 = MPI_Wtime();
    Filter* filter = new Filter(sim->da_nodes, sim->xPhys, sim->filter, sim->rmin);
    t2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Filt %f\n",t2-t1);

    // STEP 4: VISUALIZATION USING VTK
//    MPIIO* output = new MPIIO(opt->da_nodes, 3, "ux, uy, uz", 3, "x, xTilde, xPhys");
    // STEP 5: THE OPTIMIZER MMA
//    MMA*     mma;
    PetscInt itr = 0;
//    sim->AllocateMMAwithRestart(&itr, &mma); // allow for restart !
//     mma->SetAsymptotes(0.2, 0.65, 1.05);

    PetscScalar *xin;
    PetscInt     locsiz;
    ierr = VecGetLocalSize(sim->x, &locsiz);
    CHKERRQ(ierr);
    ierr = VecGetArray(sim->x, &xin);
    CHKERRQ(ierr);
//
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    int sank;
    MPI_Comm_size(PETSC_COMM_WORLD,&sank);
//
            setenv("PYTHONPATH", ".", 0);

        Py_Initialize();
        import_array();

        const int SIZE = 2;
        npy_intp dims[1]{SIZE};
        const int ND = 1;
        int *i_arr { new int[SIZE] };

        i_arr[0] = rank;
        i_arr[1] = sank;

        PyObject *iArray=PyArray_SimpleNewFromData(ND,dims,NPY_INT,reinterpret_cast<void*>(i_arr));

        // import mymodule
        const char *module_name = "pywrite";
        PyObject *pName = PyUnicode_FromString(module_name);
        PyObject *pModule = PyImport_Import(pName);
        Py_DECREF(pName);

//
    // import function without arguments
    const char *func_name2 = "read";
    PyObject *pFunc2 = PyObject_GetAttrString(pModule, func_name2);
    PyObject *pReturn2 = PyObject_CallFunctionObjArgs(pFunc2, iArray, NULL);
    PyArrayObject *np_ret2 = reinterpret_cast<PyArrayObject*>(pReturn2);

    // convert back to C++ array and print
    int len2 = PyArray_SHAPE(np_ret2)[0];
    float* c_out2;
    c_out2 = reinterpret_cast<float*>(PyArray_DATA(np_ret2));
    std::cout << "Printing output array 2 - C++" << std::endl;
    for (int i = 0; i < len2; i++){
        std::cout << c_out2[i] << ' ';
        xin[i] = c_out2[i];
    }
    ierr = VecRestoreArray(sim->x, &xin);
    CHKERRQ(ierr);

    // STEP 6: FILTER THE INITIAL DESIGN/RESTARTED DESIGN
    t1 = MPI_Wtime();
    ierr = filter->FilterProject(sim->x, sim->xTilde, sim->xPhys, sim->projectionFilter, sim->beta, sim->eta);
    CHKERRQ(ierr);
    t2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"do Filt %f\n",t2-t1);

    // STEP 7: OPTIMIZATION LOOP
    PetscScalar ch = 1.0;
//    while (itr < opt->maxItr && ch > 0.01) {
        // Update iteration counter
        itr++;

        // start timer
        t1 = MPI_Wtime();

        // Compute (a) obj+const, (b) sens, (c) obj+const+sens
        ierr = physics->ComputeObjectiveConstraintsSensitivities(&(sim->fx), &(sim->gx[0]), sim->dfdx, sim->dgdx[0],
                                                                 sim->xPhys, sim->Emin, sim->Emax, sim->penal,
                                                                 sim->volfrac);
        t2 = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"solve %f\n",t2-t1);
        CHKERRQ(ierr);

        // Compute objective scale
        if (itr == 1) {
            sim->fscale = 10.0 / sim->fx;
        }
        // Scale objectie and sens
        sim->fx = sim->fx * sim->fscale;
        VecScale(sim->dfdx, sim->fscale);

        t1 = MPI_Wtime();
        // Filter sensitivities (chainrule)
        ierr = filter->Gradients(sim->x, sim->xTilde, sim->dfdx, sim->m, sim->dgdx, sim->projectionFilter, sim->beta,
                                 sim->eta);
        CHKERRQ(ierr);
        t2 = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"do filt %f\n",t2-t1);

        // Sets outer movelimits on design variables
//        ierr = mma->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->x, opt->xmin, opt->xmax);
        CHKERRQ(ierr);

        // Update design by MMA
//        ierr = mma->Update(opt->x, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
//        CHKERRQ(ierr);

        // Inf norm on the design change
//        ch = mma->DesignChange(opt->x, opt->xold);

        // Increase beta if needed
//        PetscBool changeBeta = PETSC_FALSE;
//        if (opt->projectionFilter) {
//            changeBeta = filter->IncreaseBeta(&(opt->beta), opt->betaFinal, opt->gx[0], itr, ch);
//        }

        // Filter design field
//        ierr = filter->FilterProject(opt->x, opt->xTilde, opt->xPhys, opt->projectionFilter, opt->beta, opt->eta);
//        CHKERRQ(ierr);

        // Discreteness measure
//        PetscScalar mnd = filter->GetMND(opt->xPhys);

        // stop timer
        t2 = MPI_Wtime();

        // Print to screen
        PetscPrintf(PETSC_COMM_WORLD,
                    "It.: %i, True fx: %f, Scaled fx: %f, gx[0]: %f, ch.: %f, time: %f\n",
                    itr, sim->fx / sim->fscale, sim->fx, sim->gx[0], ch, t2 - t1);

        // Write field data: first 10 iterations and then every 20th
//        if (itr < 11 || itr % 20 == 0 || changeBeta) {
//            output->WriteVTK(physics->da_nodal, physics->GetStateField(), opt->x, opt->xTilde, opt->xPhys, itr);
//        }

        // Dump data needed for restarting code at termination
//        if (itr % 10 == 0) {
//        t1 = MPI_Wtime();
//        sim->WriteRestartFiles(&itr);
//        t2 = MPI_Wtime();
//        PetscPrintf(PETSC_COMM_WORLD,"do filt %f\n",t2-t1);
//            physics->WriteRestartFiles();
//        }
//    }
    // Write restart WriteRestartFiles
//    opt->WriteRestartFiles(&itr, mma);
//    physics->WriteRestartFiles();

    // Dump final design
//    output->WriteVTK(physics->da_nodal, physics->GetStateField(), opt->x, opt->xTilde, opt->xPhys, itr + 1);

        return 1;

        PetscScalar *df;
//        PetscInt     locsiz;
        ierr = VecGetLocalSize(sim->x, &locsiz);
        CHKERRQ(ierr);
        ierr = VecGetArray(sim->dfdx, &df);
        CHKERRQ(ierr);
//
//////
//
//        setenv("PYTHONPATH", ".", 0);

//        Py_Initialize();
//        import_array();

        const int SIZEo = locsiz+1;
        npy_intp dimso[1]{SIZEo};
//       npy_intp dims[2]{SIZE, SIZE};
        const int NDo = 1;
        float *c_arr { new float[SIZEo] };
//      long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };
     
//        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

        c_arr[0] = rank;
        for (int i = 0; i < SIZEo; i++){
            c_arr[i+1] = df[i];
        }

        printf("df %f\n",df[0]);
        printf("c %f\n",c_arr[0]);

        ierr = VecRestoreArray(sim->dfdx, &df);
        CHKERRQ(ierr);

        // Convert it to a NumPy array.
        PyObject *pArray=PyArray_SimpleNewFromData(NDo,dimso,NPY_FLOAT,reinterpret_cast<void*>(c_arr));
    

        // import function
        const char *func_name = "write";
        PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
        PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
        PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);

        Py_Finalize();

//    sim->dfdx

    // STEP 7: CLEAN UP AFTER YOURSELF
//    delete mma;
//    delete output;
    delete filter;
    delete sim;
    delete physics;

    // Finalize PETSc / MPI
    PetscFinalize();
    return 0;
}
