/*
 * A Gmsh template file for a rectangle domain.
 */
lc = 0.25;
// vertices.
Point(1) = {0,0,0,lc};
Point(2) = {4,0,0,lc};
Point(3) = {4,1,0,lc};
Point(4) = {0,1,0,lc};
// lines.
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
// surface.
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
// physics.
Physical Line("lower") = {1};
Physical Line("right") = {2};
Physical Line("upper") = {3};
Physical Line("left") = {4};
Physical Surface("domain") = {1};
// mesh
Mesh.MshFileVersion = 2.2;
Mesh.ALgorithm = 6; // Frontal-Delaunay for 2D mesh.
Mesh 2;