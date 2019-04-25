-----------------------
Compile
-----------------------
> make

-----------------------
Run 
-----------------------


To preprocess a set of edges in matrix market patern format
> ./pprocess.sh edges.dat



You can run separately
Sort :
> ./sort edges.dat

Compute H :
> ./H edges.dat

Compute H transposed and dangling node vector
> ./HTA H.mtx

Convert in AmgX binary format
> ./mtob HTA.mtx

-----------------------
Input
-----------------------
The format for sort and H is matrix market patern format
example :

%%comment
% as much comments as you want
%...
size size nonzero
a b
c d
a e
e a
.
.
.
[a-e] are in N*


The format for HTA and mtob is matrix market coordinate format
%%comment
% as much comments as you want
%...
size size nonzero
a b f
c d g
a e h
e a i
.
.
.
[a-e] are in N*
[f-i] are in R