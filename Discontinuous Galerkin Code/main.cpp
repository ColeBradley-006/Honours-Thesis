#include <iostream>
#include<stdlib.h>
#include<cmath>
#include<Eigen/Dense>
#include<Eigen/Sparse>
using namespace std;
using namespace Eigen;



//Good with no bug
VectorXf jacobiP(int N, float a, float b, VectorXf x){
    
    VectorXf xp = x;
    int dims = xp.size();
    if (dims == 1){
        xp.transposeInPlace();
        dims = xp.size();
    }
    float gamma0 = pow(2.0,a+b+1)/(a+b+1)*tgamma(a+1)*tgamma(b+1)/tgamma(a+b+1);
    MatrixXf PL(N+1,dims);
    PL.row(0) = VectorXf::Ones(dims).transpose() / sqrt(gamma0);

    if (N == 0) {
        return PL.row(0).transpose();
    }

    float gamma1 = (a+1)*(b+1)/(a+b+3)* gamma0;
    PL.row(1) = ((a+b+2)*xp.array() / 2 + (a-b)/2).transpose() /sqrt(gamma1);

    if (N==1){
        return PL.row(1).transpose();
    }

    float aold = 2/(2+a+b)*sqrt((a+1)*(b+1)/(a+b+3));
    for (int i = 0; i < N - 1; i++) {
        float h1 = 2*i+a+b;
        float anew = 2.0/(h1+2)*sqrt((i+1)*(i+1+a+b)*(i+1+a)*(i+1+b)/(h1+1)/(h1+3));
        float bnew;
        if (a*a-b*b != 0 && h1!= 0){
            float bnew = -(a*a-b*b)/h1/(h1+2); 
        }
        else{
            float bnew = 0;
        }
        VectorXf right = (xp.array() - bnew).matrix();
        VectorXf left = -aold*PL.row(i);
        VectorXf right2 = right.cwiseProduct(PL.row(i+1).transpose());
        PL.row(i+2) = 1.0 / anew * (left + right2);
        aold = anew;
    }
    return PL.row(N).transpose();
}

//Complete with no bug
VectorXf gradJacobiP(int N, float a, float b, VectorXf x){
    VectorXf dP(x.size());
    if (N == 0){
        dP.setZero();
    }
    else{
        dP = sqrt(N*(N+a+b+1)) * jacobiP(N-1,a+1,b+1,x);
    }
    return dP;
}

//Complete with no bug
tuple<VectorXf,VectorXf> jacobiGQ(int N, float a, float b){
    VectorXf x(N), w(N);

    if (N == 0)
    {
        x(0) = (a-b) / (a+b+2);
        w(0) = 2;
        return tuple<VectorXf, VectorXf>{x,w};
    }

    // Form symmetric matrix from recurrence.
    MatrixXf J(N + 1, N + 1);
    J.setZero();
    VectorXf h1 = (2 * VectorXf::LinSpaced(N + 1, 0, N)).array() + a + b;
    J.diagonal() = (-0.5 * (a*a-b*b) / (h1.array() + 2) / h1.array());
    J.diagonal(1) = 2 / (h1.head(N).array() + 2) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N)).array() *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + a+b).array()) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + a).array()) *
        ((1.0 * VectorXf::LinSpaced(N, 1, N).array() + b).array()) /
        (h1.head(N).array() + 1) / (h1.head(N).array() + 3)).sqrt();

    if (a+b< 10 * std::numeric_limits<float>::epsilon())
    {
        J(0, 0) = 0.0;
    }
    MatrixXf JT = J.transpose();
    J = J + JT;

    // Compute quadrature by eigenvalue solve
    SelfAdjointEigenSolver<MatrixXf> eigensolver(J);
    x = eigensolver.eigenvalues();
    VectorXf v1 = eigensolver.eigenvectors().col(0);
    w = v1.array().square() * std::pow(2, a + b + 1) /
        (a + b + 1) * std::tgamma(a + 1) *
        std::tgamma(b + 1) / std::tgamma(a + b + 1);
    return tuple<VectorXf,VectorXf>{x,w};
}

//Complete with no bug
VectorXf jacobiGL(int N, float a, float b){
    VectorXf x(N+1);
    x.setZero();
    if (N==1){
        x(0) = -1.0;
        x(1) = 1.0;
        return x;
    }
    tuple<VectorXf,VectorXf> gq = jacobiGQ(N, a + 1, b+ 1);

    VectorXf quads = get<0>(gq);
    x(0) = -1.0;
    for (int i = 1; i < N; i++){
        x(i) = quads(i-1);
    }
    x(N) = 1.0;
    return x;
}

//Complete with no bug
MatrixXf vandermonde(int N, VectorXf x){
    int rows = x.size();
    MatrixXf V1D(rows, N+1);
    for (int j = 0; j <= N; j++){
        VectorXf jac = jacobiP(j,0,0,x);
        V1D.col(j) = jac;
    }
    return V1D;
} 


//Complete with no bug
MatrixXf gradVandermonde(int N, VectorXf x){
    MatrixXf DV1D(x.size(), N+1);

    for (int j = 0; j <= N; j++){
        DV1D.col(j) = gradJacobiP(j,0,0,x);
    }
    return DV1D;
}


//Complete with no bug
MatrixXf dMatrix(int N, VectorXf x, MatrixXf V){
    MatrixXf Vr = gradVandermonde(N, x);
    return Vr * V.inverse();
}


//Complete with no bug
MatrixXf normals1D(int Nfp, int Nfaces, int K){
    MatrixXf nx(Nfp*Nfaces, K);
    for (int i = 0; i < Nfp * Nfaces; i++){
        nx(0,i) = -1.0;
        nx(1,i) = 1.0;
    }
    return nx;
}


//Complete with no bug
tuple<int, VectorXf, int, MatrixXd> meshGen(float xMin, float xMax, int K){
    int Nv = K + 1;
    VectorXf VX(Nv);

    for (int i = 0; i < Nv; i++){
        VX(i) = (xMax-xMin) * (i-1)/(Nv-1) + xMin;
    }

    MatrixXd EToV(K,2);
    for (int k = 0; k < K; k++){
        EToV(k,0) = k;
        EToV(k,1) = k+1;
    }

    return tuple<int, VectorXf, int, MatrixXd>{Nv,VX,K,EToV};
}



//Good, no bugs
tuple<MatrixXf,MatrixXf> geometricFactors(MatrixXf x, MatrixXf Dr){
    MatrixXf J = Dr*x;
    MatrixXf rx = 1.0/J.array();
    return tuple<MatrixXf,MatrixXf>{J,rx.matrix()};
}

//The rest of the function are written into the main script
int main() {
    
    int N = 8;
    tuple<int, VectorXf, int, MatrixXd> mesh = meshGen(0.0,40.0,80);
    int Nv = get<0>(mesh);
    int K = get<2>(mesh);
    VectorXf VX = get<1>(mesh);
    MatrixXd EToV = get<3>(mesh);
    
    VectorXf rk4a(5);
    VectorXf rk4b(5);
    VectorXf rk4c(5);
    rk4a << 0.0,
    -567301805773.0/1357537059087.0,
    -2404267990393.0/2016746695238.0,
    -3550918686646.0/2091501179385.0,
    -1275806237668.0/842570457699.0;
    rk4b << 1432997174477.0/9575080441755.0,
    5161836677717.0/13612068292357.0,
    1720146321549.0/2090206949498.0,
    3134564353537.0/4481467310338.0,
    2277821191437.0/14882151754819.0;
    rk4c << 0.0,
    1432997174477.0/9575080441755.0,
    2526269341429.0/6820363962896.0,
    2006345519317.0/3224310063776.0,
    2802321613138.0/2924317926251.0;

    //The following section of the code is the startup code
    float NODETOL = 1e-10;
    int Np = N + 1;
    int Nfp = 1;
    int Nfaces = 2;
    int TotalFaces = Nfaces * K;
    
    
    VectorXf r =jacobiGL(N,0,0);
    MatrixXf V = vandermonde(N, r);
    MatrixXf invV = V.inverse();
    MatrixXf Dr = dMatrix(N,r,V);
    
    //The following Section is the Lift code:
    MatrixXf Emat(Np, Nfaces*Nfp);
    Emat.setZero();
    Emat(0,0) = 1.0;
    Emat(Np - 1,1) = 1.0;
    MatrixXf vt = V.transpose();
    MatrixXf lift = V * (vt * Emat);
    //The lift code is finished

    VectorXf va = EToV.col(0).transpose();
    VectorXf vb = EToV.col(1).transpose();


    MatrixXf x(Np,K);
    for (int i = 0; i < Np; i++) {
        
        for (int j = 0; j < K; j++){

            int aLoc = va(j);
            int bLoc = vb(j);
            float rValue = r(i) + 1.0;
            x(i,j) = VX(aLoc) + 0.5 * rValue * (VX(bLoc)-VX(aLoc));
        }
    }

    
    tuple<MatrixXf,MatrixXf> geom = geometricFactors(x, Dr);
    
    std::vector<int> fmask1, fmask2;
    for (int i = 0; i < Np; i++) {
        if (abs(r(i) + 1) < NODETOL) {
            fmask1.push_back(i);
        }
        if (abs(r(i) - 1) < NODETOL) {
            fmask2.push_back(i);
        }
    }
    
    MatrixXf Fmask(fmask1.size() + fmask2.size(), 1);
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
        Fx.row(i) = x.row(Fmask(i));
    }

    MatrixXf nx = normals1D(Nfp, Nfaces, K);
    
    MatrixXf J = get<1>(geom);
    MatrixXf Fscale(J.size(),K);
    for (int i = 0; i < Fmask.size(); i ++){
        Fscale.row(i) = 1 / (J.row(Fmask(i)).array());
    }

    //Below is the connect1D code

    Vector2d vn(0,1);
    SparseMatrix<int> SpFToV(TotalFaces, Nv);
    SpFToV.reserve(2 * TotalFaces);
    int sk = 0;
    for (int k = 0; k < K; k++) {
        for (int face = 0; face < Nfaces; face++) {
            int loc1 = vn(face);
            int col = EToV(k,loc1);
            SpFToV.coeffRef(sk, col) = 1;
            sk++;
        }
    }
    Eigen::SparseMatrix<int> sparseIdentity(TotalFaces,TotalFaces);
    sparseIdentity.setIdentity();
    SparseMatrix<int> SpFToF = SpFToV * SpFToV.transpose() - sparseIdentity;

    // Find complete face to face connections
    std::vector<int> faces1, faces2;
    for (int k = 0; k < SpFToF.outerSize(); k++) {
        for (SparseMatrix<int>::InnerIterator it(SpFToF, k); it; ++it) {
            if (it.value() == 1) {
                faces1.push_back(k);
                faces2.push_back(it.col());
            }
        }
    }

    // Convert face global number to element and face numbers
    VectorXi element1 = (VectorXi::Map(&faces1[0], faces1.size()) / Nfaces).array() + 1;
    VectorXi face1 = ((VectorXi::Map(&faces1[0], faces1.size())).array() % Nfaces) + 1;
    VectorXi element2 = (VectorXi::Map(&faces2[0], faces2.size()) / Nfaces).array() + 1;
    VectorXi face2 = (VectorXi::Map(&faces2[0], faces2.size()) % Nfaces).array() + 1;

    // Rearrange into Nelements x Nfaces sized arrays
    MatrixXi ind(K, Nfaces);
    ind.col(0) = element1 - 1;
    ind.col(1) = face1 - 1;
    EToE.resize(K, Nfaces);
    EToE.setConstant(-1);
    EToE.array().col(face1 - 1) = element2.array();
    EToF.resize(K, Nfaces);
    EToF.setConstant(-1);
    EToF.array().col(face1 - 1) = face2.array();

    return 0;
}
