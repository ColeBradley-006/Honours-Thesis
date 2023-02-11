#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace std;

class Grid{
    public:
    
    //The below creates a grid with our initial condition in place
    Grid(int meshSize){
        dx = X / meshSize;
        points = meshSize;
        for (int i = 0; i < 21; i++){
            grid[i] = new double[meshSize]{0};
        }
        for (int i = 0; i < points; i++)
        {
            double calc = 0.5 * (1.0 + tanh (250.0 * (i * dx - 20.0)));
            grid[0][i] = calc;
        }
    }

    //This function prints our grid to a text file which can then be analyzed in python
    void print(string name){
        fstream myfile;

        myfile.open(name + ".txt",fstream::out);

        for (int n = 0; n < 21; n ++){
            for (int j = 0; j < points; j++){
                myfile << grid[n][j] << "\t";
            }
            myfile<<endl;
        }
        myfile.close();
        for (int i = 0; i < 21; i++){
            delete[] grid[i];
        }
    }

    //Solves for the exact values considering the initial condition
    void exact(){
        for (int n = 1; n < 21; n++){
            for (int j = 0; j < points; j++){
                grid[n][j] = 0.5 * (1.0 + tanh (250.0 * ((j * dx - 0.5 * (n/2.0)) - 20.0)));
            }
        }
    }

    //Uses the upwind solver
    void upwind(){
        for (int n = 0; n < 20; n++)
        {
            for (int j = 1; j < points; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
    }

    //Uses the lax solver
    void lax(){
        for (int n = 0; n < 20; n ++){
            for (int j = 1; j < points; j ++){
                grid[n + 1][j] = 0.5 * (grid[n][j + 1] + grid[n][j - 1]) - dt / dx * c / 2 * (grid[n][j + 1] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][points - 1] = grid[n][points - 1];
        }
    }

    //Uses the Lax-Wendroff Solver
    void lax_wendroff(){
        for (int n = 0; n < 20; n++){
            for (int j = 1; j < points; j++){
                grid[n + 1][j] = grid[n][j] - c / 2 * dt / dx * (grid[n][j + 1] - grid[n][j - 1]) + pow(c * dt / dx, 2.0) / 2 * (grid[n][j + 1] - 2 * grid[n][j] + grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][points - 1] = grid[n][points - 1];
        }
    }

    //Uses the leapfrog Solver
    void leap_frog(){
        //For the first time step do upwind and then do leap_frog for the remainder
        for (int j = 1; j < points; j ++){
            grid[1][j] = grid[0][j] - dt / dx * c * (grid[0][j] - grid[0][j - 1]);
            }
        grid[1][0] = grid[0][0];
        for (int n = 1; n < 20; n++){
            for (int j = 1; j < points; j++){
                grid[n + 1][j] = grid[n - 1][j] - c * dt / dx * (grid[n][j + 1]- grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][points - 1] = grid[n][points - 1];   
        }
    }

    //Uses the MacCormack Solver Technique
    void maccormack(){
        //This finds the u n + 1 bar term
        for (int n = 0; n < 20; n++)
        {
            for (int j = 0; j < points; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
        //Now we find the u n + 1 term
        for (int n = 0; n < 20; n++)
        {
            for (int j = 1; j < points; j ++){
                grid[n + 1][j] = 0.5 * (grid[n + 1][j] + grid[n][j] - dt / dx * (grid[n + 1][j] - grid[n + 1][j - 1]));
            }
        }
    }

    //We calculate the error at time step of 5s
    static double error(Grid test, Grid exact, int size){
        double tot = 0;
        for (int i = 1; i < size; i++){
            double ue = exact.grid[10][i];
            double ut = test.grid[10][i];
            double error = pow(ue - ut, 2);
            tot += error;
        }
        return sqrt(tot * test.dx);
    }

    //Initializes all the grid data
    protected:
    float c = 0.5;
    float t0 = 0.0;
    float tf = 10.0;
    float X = 41.0;
    float dt = 0.5;
    int T = 20;
    int points;
    float dx;
    double** grid = new double*[21];
};

int main(){
    cout << "\tWelcome to the Linear Advection solver" << endl << endl ;
    cout << "This solver will use five different schemes to solve the linear advection equation" << endl;
    cout << "The initial condition is u = 1/2 * (1 + tanh[250(x-20)]) for x between 0 and 40" << endl << endl;
    int meshPoints[4] = {10, 20, 41, 82};
    double errors[5][4]= {0};
    for (int j = 0; j < 4; j++){
        cout << "Calculating the exact solution for the PDE..." << endl << endl;
        Grid exact_grid = Grid(meshPoints[j]);
        exact_grid.exact();
        for (int k = 0; k < 5; k ++){
            switch(k){
                case 0:{
                    cout << "Calculating the solution using upwind technique for the PDE..." << endl << endl;
                    Grid upwind_grid = Grid(meshPoints[j]);
                    upwind_grid.upwind();
                    errors[k][j] = Grid::error(upwind_grid, exact_grid, meshPoints[j]);
                    upwind_grid.print("upwind" + to_string(j));
                    break;
                }
                case 1:{
                    cout << "Calculating the solution using Lax technique for the PDE..." << endl << endl;
                    Grid lax_grid = Grid(meshPoints[j]);
                    lax_grid.lax();
                    k = 1;
                    errors[k][j] = Grid::error(lax_grid, exact_grid, meshPoints[j]);
                    lax_grid.print("lax" + to_string(j));
                    break;
                }
                case 2:{
                    cout << "Calculating the solution using Lax-Wendroff technique for the PDE..." << endl << endl;
                    Grid laxW_grid = Grid(meshPoints[j]);
                    laxW_grid.lax_wendroff();
                    k = 2;
                    errors[k][j] = Grid::error(laxW_grid, exact_grid, meshPoints[j]);
                    laxW_grid.print("laxWendroff" + to_string(j));
                    break;
                }
                case 3:{
                    cout << "Calculating the solution using Leap-Frog technique for the PDE..." << endl << endl;
                    Grid LF_grid = Grid(meshPoints[j]);
                    LF_grid.leap_frog();
                    k = 3;
                    errors[k][j] = Grid::error(LF_grid, exact_grid, meshPoints[j]);
                    LF_grid.print("LeapFrog" + to_string(j));
                    break;
                }
                case 4:{
                    cout << "Calculating the solution using MacCormack technique for the PDE..." << endl << endl;
                    Grid mac_grid = Grid(meshPoints[j]);
                    mac_grid.maccormack();
                    errors[k][j] = Grid::error(mac_grid, exact_grid, meshPoints[j]);
                    mac_grid.print("maccormack" + to_string(j));
                }
            }
        }
        exact_grid.print("exact" + to_string(j));
    }
    
    fstream myfile;
    myfile.open("errors.txt",fstream::out);
    for (int n = 0; n < 5; n ++){
        for (int j = 0; j < 4; j++){
        myfile << errors[n][j] << "\t";
        }
        myfile<<endl;
    }
    myfile.close();

    cout << "All text files created" << endl;
    return 0;
}