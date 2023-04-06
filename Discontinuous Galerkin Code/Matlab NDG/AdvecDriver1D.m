% Driver script for solving the 1D advection equations
Globals1D;
% Order of polymomials used for approximation
N = 8;
% Gener
[Nv, VX, K, EToV] = MeshGen1D(0.0,40.0,80);
% Initialize solver and construct grid and metric
StartUp1D;
% Set initial conditions
u = 0.5*(1+tanh(250*(x-20)));
% Solve Problem
FinalTime = 5;
[u] = Advec1D(u,FinalTime);

Y = u(9,:);
xStart = 0;
dx = 0.5;
N = 80;
X = xStart + (0:N-1)*dx;

Yexact = 0.5*(1.0 + tanh((X-0.5*5)-20));
plot(X,Y)
hold on
plot(X,Yexact,'Color','red')
