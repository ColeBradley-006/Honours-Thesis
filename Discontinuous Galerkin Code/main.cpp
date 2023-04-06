#include <iostream>
#include<stdlib.h>
#include<cmath>
#include<Eigen/Dense>
#include<Eigen/Sparse>
using namespace std;
using namespace Eigen;



//Good with proper outputs
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
        float h1 = 2*(i+1)+a+b;
        float anew = 2.0/(h1+2)*sqrt(((i+1)+1)*((i+1)+1+a+b)*((i+1)+1+a)*((i+1)+1+b)/(h1+1)/(h1+3));
        float bnew = 0;
        VectorXf right = (xp.array() - bnew).matrix();
        VectorXf left = -aold*PL.row(i);
        VectorXf right2 = right.cwiseProduct(PL.row(i+1).transpose());
        PL.row(i+2) = 1.0 / anew * (left + right2);
        aold = anew;
    }
    return PL.row(N).transpose();
}

//Good with proper outputs
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

//Complete, outputting the proper results
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

//Complete, outputting the proper results
VectorXf jacobiGL(int N, float a, float b){
    VectorXf x(N+1);
    x.setZero();
    if (N==1){
        x(0) = -1.0;
        x(1) = 1.0;
        return x;
    }
    tuple<VectorXf,VectorXf> gq = jacobiGQ(N-2, a + 1, b+ 1);

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
    return tuple<MatrixXf,MatrixXf>{rx.matrix(),J};
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

    VectorXd va = EToV.col(0).transpose();
    VectorXd vb = EToV.col(1).transpose();

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
    MatrixXf Fx(Fmask.size(), x.row(0).size());
    
    for (int i = 0; i < Fmask.size(); i++) {
        Fx.row(i) = x.row(Fmask(i));
    }
    
    MatrixXf nx = normals1D(Nfp, Nfaces, K);
    
    MatrixXf J = get<1>(geom);
    MatrixXf Fscale(Fmask.size(),K);
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
                faces1.push_back(it.row());
                faces2.push_back(it.col());
            }
        }
    }


    VectorXi faces1v = Eigen::Map<VectorXi,Eigen::Unaligned>(faces1.data(),faces1.size());
    VectorXi faces2v = Eigen::Map<VectorXi,Eigen::Unaligned>(faces2.data(),faces2.size());

    VectorXi element1 = (((faces1v.array() - 1) / Nfaces).floor() + 1).matrix();
    VectorXi face1 = ((faces1v.array() - (Nfaces * (faces1v.array()/ Nfaces))) + 1).matrix();
    VectorXi element2 = (((faces2v.array() - 1) / Nfaces).floor() + 1).matrix();
    VectorXi face2 = ((faces2v.array() - (Nfaces * (faces2v.array()/ Nfaces))) + 1).matrix();

    MatrixXi EToF(K,Nfaces);
    MatrixXi EToE(K,Nfaces);
    for (int k = 0; k < K; ++k){
        for (int i = 0; i < Nfaces; ++i){
            EToF(k,i) = i;
            EToE(k,i) = k;
        }
    }
    for (int i = 0; i < element1.size(); ++i){
        EToE(element1(i),face1(i)) = element2(i);
        EToF(element1(i), face1(i)) = face2(i);
    }
    cout << EToE << endl;
    cout << EToF << endl;
    //Connect1D finished
    
      //BuildMaps1D 
    int ***vmapM_3D = new int**[Nfp];
    int ***vmapP_3D = new int**[Nfp];
    for(int i = 0; i<Nfp; ++i){
        vmapM_3D[i] = new int*[Nfaces];
        vmapP_3D[i] = new int*[Nfaces];
        for(int j = 0; j<Nfaces; ++j){
        vmapM_3D[i][j] = new int[K];
        vmapP_3D[i][j] = new int[K];
        for(int l = 0; l<K;++l){
            vmapM_3D[i][j][l] = 0;
            vmapP_3D[i][j][l] = 0;
        }
        }
    }
    int **nodeids = new int*[Np];
    for(int n = 0; n<Np; ++n){
        nodeids[n] = new int[K];
    }
    sk = 0;
    for(int k = 0; k<K; ++k){
        for(int n = 0; n<Np; ++n){
        nodeids[n][k] = sk;
        ++sk;
        }
    }

    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
        for(int n = 0; n<Nfp; ++n){
            int value = Fmask(n,f);
            vmapM_3D[n][f][k] = nodeids[value][k];
        }
        }
    }
    for(int n = 0; n<Np; ++n){
        delete [] nodeids[n];
    }
    delete [] nodeids;

    double *x1 = new double[Nfp];
    double *x2 = new double[Nfp];
    //REVISAR!!!
    int k2, f2;
    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
            sk = 1;
            k2 = EToE(k,f);
            f2 = EToF(k,f);
            for(int n = 0; n<Nfp; ++n){
                x1[n] = x(vmapM_3D[n][f][k]%Np,vmapM_3D[n][f][k]/Np);
                x2[n] = x(vmapM_3D[n][f2][k2]%Np,vmapM_3D[n][f2][k2]/Np);
                sk*=(pow(x1[n]-x2[n],2)<NODETOL);
            }
            if(sk) for(int n = 0; n<Nfp; ++n) vmapP_3D[n][f][k] = vmapM_3D[n][f2][k2];
        }
    }
    delete [] x1;
    delete [] x2;
    int *vmapM = new int[Nfp*Nfaces*K];
    int *vmapP = new int[Nfp*Nfaces*K];
    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
        for(int n = 0; n<Nfp; ++n){
            vmapM[n+f*Nfp+k*Nfp*Nfaces] = vmapM_3D[n][f][k];
            vmapP[n+f*Nfp+k*Nfp*Nfaces] = vmapP_3D[n][f][k];
        }
        }
    }
    int res1 = 0;
    for(int i = 0; i<Nfp*Nfaces*K; ++i){
        res1 += (vmapM[i]==vmapP[i]);
    }
    int dimmapB = res1;
    int *vmapB = new int[res1];
    int *mapB = new int[res1];
    int i1 = 0;
    for(int i = 0; i<Nfp*Nfaces*K; ++i){
        if(vmapM[i] == vmapP[i]){
        mapB[i1] =  i;
        vmapB[i1] = vmapM[i];
        ++i1;
        }  
    }
    for(int n = 0; n<Nfp; ++n){
        for(int f=0; f<Nfaces; ++f){
        delete [] vmapM_3D[n][f];
        delete [] vmapP_3D[n][f];
        } 
        delete [] vmapM_3D[n];
        delete [] vmapP_3D[n];
    }
    delete [] vmapM_3D;
    delete [] vmapP_3D;
    int mapI = 0;
    int mapO = K*Nfaces-1;
    int vmapI = 0;
    int vmapO = K*Np-1;



    //Below We Begin the advection code
    //This will run the timesteps and solve the problem

    return 0;   
}
