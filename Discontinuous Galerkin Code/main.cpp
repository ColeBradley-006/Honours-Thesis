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

//This should work if calculations are correct
MatrixXf advecRHS(int Nfp, int Nfaces, int K, VectorXi vmapM, VectorXi vmapP, MatrixXf nx, float a, MatrixXf u, int mapI, int vmapI, int mapO, MatrixXf Dr, MatrixXf rx, MatrixXf Lift, MatrixXf Fscale){
    float alpha = 1;

    MatrixXf du(Nfp*Nfaces, K);

    for (int i = 0; i < vmapM.size(); i++){
        int rowL = vmapM(i)%i;
        int colL = floor(vmapM(i)/i);
        int rowR = vmapP(i)%i;
        int colR = floor(vmapP(i)/i);
        float coeff = u(rowL, colL) - u(rowR, colR);
        du.col(i) = ((coeff * (a*nx.col(i).array() - (1-alpha)*abs(a*nx.col(i).array())))/2).matrix();
    }

    //Now boundary conditions
    int uin = 0;
    du(0,0) = (u(0,0) - uin) * (a*nx(0,0) - (1-alpha)*abs(a*nx(0,0)))/2;
    int row = mapO%Nfaces;
    int col = floor(mapO/Nfaces);
    du(row, col) = 0.0;
    du = (-a*rx.array()*((Dr*u).array())).matrix() + Lift*((Fscale.array()*du.array()).matrix());
    return du;
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
    MatrixXf rx = get<0>(geom);
    
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
    SparseMatrix<int> SpFToFR = SpFToV * SpFToV.transpose() - sparseIdentity;

    // Find complete face to face connections
    std::vector<int> faces1, faces2;
    for (int k = 0; k < SpFToFR.outerSize(); k++) {
        for (SparseMatrix<int>::InnerIterator it(SpFToFR, k); it; ++it) {
            if (it.value() == 1) {
                faces1.push_back(it.row());
                faces2.push_back(it.col());
            }
        }
    }

    VectorXi faces1v = Eigen::Map<VectorXi,Eigen::Unaligned>(faces1.data(),faces1.size());
    VectorXi faces2v = Eigen::Map<VectorXi,Eigen::Unaligned>(faces2.data(),faces2.size());

    VectorXi element1 = ((faces1v.array() / Nfaces).floor()).matrix();
    VectorXi face1 = (faces1v.array() - (Nfaces * (faces1v.array()/ Nfaces))).matrix();
    VectorXi element2 = ((faces2v.array() / Nfaces).floor()).matrix();
    VectorXi face2 = (faces2v.array() - (Nfaces * (faces2v.array()/ Nfaces))).matrix();
    MatrixXi EToF(K,Nfaces);
    MatrixXi EToE(K,Nfaces);

    for (int k = 0; k < K; ++k){
        for (int i = 0; i < Nfaces; ++i){
            EToF(k,i) = i;
            EToE(k,i) = k;
        }
    }

    for (int i = 1; i < K; ++i){
        EToE(i,0) = element2(2*(i-1));
        EToF(i,0) = face2(2*(i-1));
    }

    for (int i = 0; i < K-1; ++i){
        EToE(i,1) = element2(2*i+1);
        EToF(i,1) = face2(2*i+1);
    }

    //Connect1D finished
    //BuildMaps1D 
    
    MatrixXi nodeids(Np,K);
    int number = 0;
    for (int j = 0; j < K; ++j){
        for (int i = 0; i < Np; ++i){
            nodeids(i,j) = number;
            number += 1;
        }
    }
    
    int **vmapM_3D = new int*[K];
    int **vmapP_3D = new int*[K]; 
    for(int i = 0; i<K; ++i){
        vmapM_3D[i] = new int[Nfaces];
        vmapP_3D[i] = new int[Nfaces];
        for(int j = 0; j<Nfaces; ++j){
            vmapM_3D[i][j] = 0;
            vmapP_3D[i][j] = 0;
        }
    }
    
    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
            int row = Fmask(f);
            int node = nodeids(row, k);
            vmapM_3D[k][f] = node;
        }
    }
    cout << "EToE" << endl << EToE << endl;
    cout << "EToF" << endl << EToF << endl;
    int k2, f2;
    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
            k2 = EToE(k,f);
            f2 = EToF(k,f);
            
            int vidM = vmapM_3D[k][f];
            int vidP = vmapM_3D[k2][f2];
            int col1 = floor(vidM/Np);
            int col2 = floor(vidP/Np);
            float x1 = x(vidM%Np, col1);
            float x2 = x(vidP%Np, col2);

            double D = pow((x1-x2),2);
            if (D < NODETOL){
                vmapP_3D[k][f] = vmapM_3D[k2][f2];
            }
        }
    }

    VectorXi vmapP(Nfp*Nfaces*K);
    VectorXi vmapM(Nfp*Nfaces*K);

    int location = 0;
    for(int k = 0; k<K; ++k){
        for(int f = 0; f<Nfaces; ++f){
            vmapP(location) = vmapP_3D[k][f];
            vmapM(location) = vmapM_3D[k][f];
            location += 1;
        }
    }
    
    //Now find boundary nodes
    std::vector<int> boundaries;

    for (int i =0; i < vmapP.size(); i ++){
        if(vmapP(i) = vmapM(i)){
            boundaries.push_back(i);
        }
    }

    VectorXi vmapB(boundaries.size());

    for (int i = 0; i < boundaries.size(); i++){
        vmapB(i) = vmapM(boundaries.at(i));
    }
    
    //Must delete the 3D dynamic memory arrays to avoid mem leak!!
    for(int n = 0; n<K; ++n){
        delete [] vmapM_3D[n];
        delete [] vmapP_3D[n];
    }
    delete [] vmapM_3D;
    delete [] vmapP_3D;

    int mapI = 0;
    int mapO = K*Nfaces-1;
    int vmapI = 0;
    int vmapO = K*Np-1;
    cout << "VmapM" << endl << vmapM << endl;
    cout << "VmapP" << endl << vmapP << endl;
    cout << "VmapB" << endl << vmapB << endl;
    //Build maps code complete!!!
    
    //Below We Begin the advection code
    float FinalTime = 5;
    MatrixXf u = (0.5*(1+tanh(250*(x.array()-20)))).matrix();

    float time = 0;
    MatrixXf residual = MatrixXf::Zero(Np,K);
    float dt = 0.001;
    int Nsteps = ceil(FinalTime/dt);
    float a = 0.5;

    for (int t=0; t < Nsteps; t++){
        for (int i = 0; i < 5; i++){
            float localTime = time + rk4c(i) * dt;
            MatrixXf rhsu = advecRHS(Nfp, Nfaces, K, vmapM, vmapP, nx, a, u, mapI, vmapI, mapO, Dr, rx, lift, Fscale);
            MatrixXf result = rk4a(i)*result + dt*rhsu;
            u = u + rk4b(i)*result;
        }
        time = time + dt;
    }

    return 0;   
}
