====
modmesh: modules to solve conservation laws with unstructured meshes
====

modmesh is a modularized code implementing `the space-time conservation element
and solution element (CESE) method
<https://yyc.solvcon.net/en/latest/cese/index.html>`__ based on unstructured
meshes of mixed element to solve conservation laws. It is developed by using
C++ and Python to provide:

1. Contiguous buffers and multi-dimensional arrays.
2. One-dimensional solvers for demonstrating the CESE method.
3. (To be ported from https://github.com/solvcon/solvcon) unstructured meshes
   of mixed elements for solving conservation laws by using the CESE method in
   two- and three-dimensional space.
4. (To be developed) two- and three-dimensional body mesh generation.
5. (Under development) an integrated runtime profiler.
6. A graphical user interface (GUI) application based on Qt for the spatial data
   and analysis.

An experimental Windows binary (portable) can be downloaded from the `devbuild
Github Action
<https://github.com/solvcon/modmesh/actions/workflows/devbuild.yml?query=event%3Aschedule+is%3Asuccess+branch%3Amaster>`__.
Click the Windows release run and scroll down to the "artifacts" section to
download the zip file (login to `GitHub <https://github.com/>`__ is required).
A direct download link can be found in https://doc.solvcon.net/.

Refenreces
==========

* The numerical notes: https://github.com/solvcon/mmnote.

.. vim: set ft=rst ff=unix tw=79: