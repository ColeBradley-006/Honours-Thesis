#include <iostream>
#include<stdlib.h>
#include<cmath>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;



//Complete
VectorXf jacobiP(int N, double a, double b, VectorXf x){
    
    VectorXf xp = x;
    int dims = xp.size();
    if (dims == 1){
        xp.transposeInPlace();
        dims = xp.size();
    }
    double gamma0 = pow(2.0,a+b+1)/(a+b+1)*tgamma(a+1)*tgamma(b+1)/tgamma(a+b+1);
    Matrix2f PL(N+1,dims);
    PL.row(0) = VectorXf::Ones(dims).transpose() / sqrt(gamma0);

    if (N == 0) {
        return PL.row(0).transpose();
    }

    double gamma1 = (a+1)*(b+1)/(a+b+3)* gamma0;
    PL.row(1) = ((a+b+2)*xp.array() / 2 + (a-b)/2).transpose() /sqrt(gamma1);

    if (N==1){
        return PL.row(1).transpose();
    }

    double aold = 2/(2+a+b)*sqrt((a+1)*(b+1)/(a+b+3));
    for (int i = 0; i < N - 1; i++) {
        double h1 = 2*i+a+b;
        double anew = 2.0/(h1+2)*sqrt((i+1)*(i+1+a+b)*(i+1+a)*(i+1+b)/(h1+1)/(h1+3));
        double bnew = -(a*a-b*b)/h1/(h1+2);
        PL.row(i+2) = 1.0 / anew * (-aold * PL.row(i) + (xp.array() - bnew).matrix() * PL.row(i + 1));
        aold = anew;
    }
    return PL.row(N).transpose();
}

//Complete
VectorXf gradJacobiP(int N, double a, double b, VectorXf x){
    VectorXf dP(x.size());
    if (N == 0){
        dP.setZero();
    }
    else{
        dP = sqrt(N*(N+a+b+1)) * jacobiP(N-1,a+1,b+1,x);
    }
    return dP;
}

//Complete
tuple<VectorXf,VectorXf> jacobiGQ(int N, double a, double b){
    VectorXf x(N), w(N);

    if (N == 0)
    {
        x(0) = (a-b) / (a+b+2);
        w(0) = 2;
        return tuple<VectorXf, VectorXf>{x,w};
    }

    // Form symmetric matrix from recurrence.
    Matrix2f J(N + 1, N + 1);
    VectorXf h1 = 2 * VectorXf::LinSpaced(N + 1, 0, N).array() + a + b;
    J = J.diagonal(-1) - J.diagonal(1);
    J.diagonal(0).array() = -0.5 * (a*a-b*b) / (h1.array() + 2) / h1.array();
    J.diagonal(1).array() += 2 / (h1.head(N).array() + 2) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N)).array() *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + a+b).array()) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + a).array()) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + b).array()) /
        (h1.head(N).array() + 1) / (h1.head(N).array() + 3)).sqrt();

    if (a+b< 10 * std::numeric_limits<double>::epsilon())
    {
        J(0, 0) = 0.0;
    }
    J = J + J.transpose();

    // Compute quadrature by eigenvalue solve
    SelfAdjointEigenSolver<Matrix2f> eigensolver(J);
    x = eigensolver.eigenvalues();
    VectorXf v1 = eigensolver.eigenvectors().col(0);
    w = v1.array().square() * std::pow(2, a + b + 1) /
        (a + b + 1) * std::tgamma(a + 1) *
        std::tgamma(b + 1) / std::tgamma(a + b + 1);
    return tuple<VectorXf,VectorXf>{x,w};
}

//Complete
VectorXf jacobiGL(int N, double a, double b){
    VectorXf x = Matrix2f::Zero(N+1,1);

    if (N==1){
        x(0) = -1.0;
        x(1) = 1.0;
        return x;
    }
    tuple<VectorXf,VectorXf> gq = jacobiGQ(N, a + 1, b+ 1);

    VectorXf quads = get<0>(gq);
    x(0) = -1.0;
    x.tail(N-1) = quads;
    x(N) = 1.0;
    return x;
}

//Complete
Matrix2f vandermonde(int N, VectorXf x){
    Matrix2f V1D(x.size(), N+1);

    for (int j = 0; j <= N; j++){
        V1D.col(j) = jacobiP(j,0,0,x);
    }
    return V1D;
}

//Complete
Matrix2f gradVandermonde(int N, VectorXf x){
    Matrix2f DV1D(x.size(), N+1);

    for (int j = 0; j <= N; j++){
        DV1D.col(j) = gradJacobiP(j,0,0,x);
    }
    return DV1D;
}

//Complete
Matrix2f dMatrix(int N, VectorXf x, Matrix2f V){
    Matrix2f Vr = gradVandermonde(N, x);
    return Vr * V.inverse();
}

//Complete
VectorXf normals1D(int Nfp, int Nfaces, int K){
    Matrix2f nx(Nfp*Nfaces, K);
    nx.row(0).setConstant(-1.0);
    nx.row(1).setConstant(1.0);
    return nx;
}

//Complete
tuple<int, VectorXf, int, VectorXf> meshGen(int xMin, int xMax, int K){
    int Nv = K + 1;
    VectorXf VX(Nv);

    for (int i = 0; i < Nv; i++){
        VX(i) = (xMax-xMin) * (i-1)/(Nv-1) + xMin;
    }

    Matrix2f EToV = Matrix2f::Zero(K,2);
    for (int k = 0; k < K; k++){
        EToV(k,1) = k;
        EToV(k,2) = k+1;
    }

    return tuple<int, VectorXf, int, VectorXf>{Nv,VX,K,EToV};
}

//Complete
tuple<Matrix2f,Matrix2f> geometricFactors(Matrix2f x, Matrix2f Dr){
    VectorXf J = Dr*x;
    VectorXf rx = J.inverse();
    return tuple<Matrix2f,Matrix2f>{J,rx};
}

//The rest of the function are written into the main script
int main() {
    int N = 8;
    tuple<int, VectorXf, int, VectorXf> mesh = meshGen(0.0,40.0,80);
    int Nv = get<0>(mesh);
    int K = get<2>(mesh);
    VectorXf VX = get<1>(mesh);
    VectorXf EToV = get<3>(mesh);
    VectorXf rk4a (0.0,
    -567301805773.0/1357537059087.0,
    -2404267990393.0/2016746695238.0,
    -3550918686646.0/2091501179385.0,
    -1275806237668.0/842570457699.0);
    VectorXf rk4b (1432997174477.0/9575080441755.0,
    5161836677717.0/13612068292357.0,
    1720146321549.0/2090206949498.0,
    3134564353537.0/4481467310338.0,
    2277821191437.0/14882151754819.0);
    VectorXf rk4c (0.0,
    1432997174477.0/9575080441755.0,
    2526269341429.0/6820363962896.0,
    2006345519317.0/3224310063776.0,
    2802321613138.0/2924317926251.0);

    //The following section of the code is the startup code
    float NODETOL = 1e-10;
    int Np = N + 1;
    int Nfp = 1;
    int Nfaces = 2;
    VectorXf r =jacobiGL(0,0,N);
    Matrix2f V = vandermonde(N, r);
    Matrix2f invV = V.inverse();
    Matrix2f Dr = dMatrix(N,r,V);
    
    //The following Section is the Lift code:
    Matrix2f Emat = Matrix2f::Zero(Np, Nfaces*Nfp);
    Emat(0,0) = 1.0;
    Emat(Np - 1,1) = 1.0;
    Matrix2f vt = V.transpose();
    Matrix2f lift = V * (vt * Emat);
    //The lift code is finished

    VectorXf va = EToV.col(0).transpose();
    VectorXf vb = EToV.col(1).transpose();

    Matrix2f x(Np,K);
    for (int i = 0; i < EToV.rows(); i++) {
        for (int j = 0; j < vb.size(); j++){
            int aLoc = va(j);
            int bloc = vb(j);
            float rValue = r(i) + 1.0;
            x(i,j) = VX(aLoc) + 0.5 * rValue * (VX(bloc)-VX(bloc));
        }
    }
    tuple<Matrix2f,Matrix2f> geom = geometricFactors(x, Dr);

    std::vector<int> fmask1, fmask2;
    for (int i = 0; i < Np; i++) {
        if (abs(r(i) + 1) < NODETOL) {
            fmask1.push_back(i);
        }
        if (abs(r(i) - 1) < NODETOL) {
            fmask2.push_back(i);
        }
    }
    Matrix2f Fmask(fmask1.size() + fmask2.size(), 1);
    int idx = 0;
    for (int i = 0; i < fmask1.size(); i++) {
        Fmask(idx) = fmask1[i];
        idx++;
    }
    for (int i = 0; i < fmask2.size(); i++) {
        Fmask(idx) = fmask2[i];
        idx++;
    }
    Fmask.transposeInPlace();
    MatrixXd Fx(Fmask.size(), 1);
    for (int i = 0; i < Fmask.size(); i++) {
        cout << x(Fmask(i));
    }
    
    VectorXf nx = normals1D(Nfp, Nfaces, K);

    Matrix2f Fscale = get<1>(geom)(Fmask, all).inverse();

    
    VectorXf u = 0.5*(1+tanh(250*(x.array()-20)));



    return 0;
}
