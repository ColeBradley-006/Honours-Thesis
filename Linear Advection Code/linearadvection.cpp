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

        for (int n = 0; n < 10; n ++){
            for (int j = 0; j < 41; j++){
                myfile << grid[n][j] << "\t";
            }
            myfile<<endl;
        }
        myfile.close();
    }

    //Solves for the exact values considering the initial condition
    void exact(){
        for (int n = 1; n < 10; n++){
            for (int j = 0; j < 41; j++){
                grid[n][j] = 0.5 * (1.0 + tanh (250.0 * ((j - 0.5 * n) - 20.0)));
            }
        }
    }

    //Uses the upwind solver
    void upwind(){
        for (int n = 0; n < 9; n++)
        {
            for (int j = 1; j < 41; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
    }

    //Uses the lax solver
    void lax(){
        for (int n = 0; n < 9; n ++){
            for (int j = 1; j < 40; j ++){
                grid[n + 1][j] = 0.5 * (grid[n][j + 1] + grid[n][j - 1]) - dt / dx * c / 2 * (grid[n][j + 1] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][40] = grid[n][40];
        }
    }

    //Uses the Lax-Wendroff Solver
    void lax_wendroff(){
        for (int n = 0; n < 9; n++){
            for (int j = 1; j < 40; j++){
                grid[n + 1][j] = grid[n][j] - c / 2 * dt / dx * (grid[n][j + 1] - grid[n][j - 1]) + pow(c * dt / dx, 2.0) / 2 * (grid[n][j + 1] - 2 * grid[n][j] + grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
            grid[n + 1][40] = grid[n][40];
        }
    }

    //Uses the leapfrog Solver
    void leap_frog(){
    }

    //Uses the MacCormack Solver Technique
    void maccormack(){
        //This finds the u n + 1 bar term
        for (int n = 0; n < 9; n++)
        {
            for (int j = 0; j < 40; j ++){
                grid[n + 1][j] = grid[n][j] - dt / dx * c * (grid[n][j] - grid[n][j - 1]);
            }
            grid[n + 1][0] = grid[n][0];
        }
        //Now we find the u n + 1 term
        for (int n = 0; n < 9; n++)
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
    float dt = 1.0;
    int T = 10;
    double grid[10][41] = {};
};


int main(){
    cout << "\tWelcome to the Linear Advection solver" << endl << endl ;
    cout << "This solver will use five different schemes to solve the linear advection equation" << endl;

    cout << "The initial condition is u = 1/2 * (1 + tanh[250(x-20)]) for x between 0 and 40" << endl << endl;
    
    string solvers[6] = {"exact", "upwind", "lax", "laxWendroff", "LeapFrog", "maccormack"};
    for (int i = 0; i < 6; i ++){
        switch(i){
            case 0: {
                Grid current = Grid();
                current.exact();
                current.print(solvers[i]);
                break;
            }
            case 1:{
                Grid current = Grid();
                current.upwind();
                current.print(solvers[i]);
                break;
            }
            case 2:{
                Grid current = Grid();
                current.lax();
                current.print(solvers[i]);
                break;
            }
            case 3:{
                Grid current = Grid();
                current.lax_wendroff();
                current.print(solvers[i]);
                break;
            }
            case 4:{
                Grid current = Grid();
                current.leap_frog();
                current.print(solvers[i]);
                break;
            }
            case 5:{
                Grid current = Grid();
                current.maccormack();
                current.print(solvers[i]);
                break;
            }
        }
    }

    cout << "Did this work?";
    return 0;
}