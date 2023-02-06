#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

class Grid{
    public:
    
    //The below creates a grid with our initial condition in place
    Grid(){
        for (int i = 0; i < X; i++)
        {
           grid[0][i] = 0.5 * (1.0 + tanh (250.0 * (i - 20.0)));
        }
    }

    //This function prints our grid to a text file which can then be analyzed in python
    void print(string name){
        fstream myfile;

        myfile.open(name + ".txt",fstream::out);

        for (int n = 0; n < 21; n ++){
            for (int j = 0; j < 41; j++){
                myfile << grid[n][j] << "\t";
            }
            myfile<<endl;
        }
        myfile.close();
    }

    //Solves for the exact values considering the initial condition
    void exact(){
        for (int n = 1; n < 21; n++){
            for (int j = 0; j < 41; j++){
                grid[n][j] = 0.5 * (1.0 + tanh (250.0 * ((j - 0.5 * (n/2.0)) - 20.0)));
            }
        }
    }

    //Uses the upwind solver
    void upwind(){
        for (int n = 0; n < 21; n++)
        {
            for (int j = 1; j < 41; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
    }

    //Uses the lax solver
    void lax(){
        for (int n = 0; n < 21; n ++){
            for (int j = 1; j < 41; j ++){
                grid[n + 1][j] = 0.5 * (grid[n][j + 1] + grid[n][j - 1]) - dt / dx * c / 2 * (grid[n][j + 1] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][40] = grid[n][40];
        }
    }

    //Uses the Lax-Wendroff Solver
    void lax_wendroff(){
        for (int n = 0; n < 21; n++){
            for (int j = 1; j < 41; j++){
                grid[n + 1][j] = grid[n][j] - c / 2 * dt / dx * (grid[n][j + 1] - grid[n][j - 1]) + pow(c * dt / dx, 2.0) / 2 * (grid[n][j + 1] - 2 * grid[n][j] + grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][40] = grid[n][40];
        }
    }

    //Uses the leapfrog Solver
    void leap_frog(){
        //For the first time step do upwind and then do leap_frog for the remainder
        for (int j = 1; j < 41; j ++){
            grid[1][j] = grid[0][j] - dt / dx * c * (grid[0][j] - grid[0][j - 1]);
            }
        grid[1][0] = grid[0][0];
        for (int n = 1; n < 21; n++){
            for (int j = 1; j < 41; j++){
                grid[n + 1][j] = grid[n - 1][j] - c * dt / dx * (grid[n][j + 1]- grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][40] = grid[n][40];   
        }
    }

    //Uses the MacCormack Solver Technique
    void maccormack(){
        //This finds the u n + 1 bar term
        for (int n = 0; n < 21; n++)
        {
            for (int j = 0; j < 41; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
        //Now we find the u n + 1 term
        for (int n = 0; n < 21; n++)
        {
            for (int j = 1; j < 41; j ++){
                grid[n + 1][j] = 0.5 * (grid[n + 1][j] + grid[n][j] - dt / dx * (grid[n + 1][j] - grid[n + 1][j - 1]));
            }
        }
    }

    //Initializes all the grid data
    protected:
    float c = 0.5;
    float t0 = 0.0;
    float tf = 10.0;
    float dx = 1.0;
    int X = 41;
    float dt = 0.5;
    int T = 20;
    double grid[21][41] = {};
};

int main(){
    cout << "\tWelcome to the Linear Advection solver" << endl << endl ;
    cout << "This solver will use five different schemes to solve the linear advection equation" << endl;
    cout << "The initial condition is u = 1/2 * (1 + tanh[250(x-20)]) for x between 0 and 40" << endl << endl;
    for (int k = 0; k < 6; k ++){
        switch(k){
            case 0: {
                cout << "Calculating the exact solution for the PDE..." << endl << endl;
                Grid exact_grid = Grid();
                exact_grid.exact();
                exact_grid.print("exact");
                break;
            }
            case 1:{
                cout << "Calculating the solution using upwind technique for the PDE..." << endl << endl;
                Grid upwind_grid = Grid();
                upwind_grid.upwind();
                k = 1;
                upwind_grid.print("upwind");
                break;
            }
            case 2:{
                cout << "Calculating the solution using Lax technique for the PDE..." << endl << endl;
                Grid lax_grid = Grid();
                lax_grid.lax();
                k = 2;
                lax_grid.print("lax");
                break;
            }
            case 3:{
                cout << "Calculating the solution using Lax-Wendroff technique for the PDE..." << endl << endl;
                Grid laxW_grid = Grid();
                laxW_grid.lax_wendroff();
                k = 3;
                laxW_grid.print("laxWendroff");
                break;
            }
            case 4:{
                cout << "Calculating the solution using Leap-Frog technique for the PDE..." << endl << endl;
                Grid LF_grid = Grid();
                LF_grid.leap_frog();
                k = 4;
                LF_grid.print("LeapFrog");
                break;
            }
            case 5:{
                cout << "Calculating the solution using MacCormack technique for the PDE..." << endl << endl;
                Grid mac_grid = Grid();
                mac_grid.maccormack();
                mac_grid.print("maccormack");
            }
        }
    }
    cout << "All text files created" << endl;
    return 0;
    }