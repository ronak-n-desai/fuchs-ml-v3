#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <math.h>

using namespace std;

int main()
{
    auto start = std::chrono::system_clock::now();
    cout << "Starting" << endl;
    
    // variable declaration
    int maxFocalOffset = 30;
    int minFocalOffset = -maxFocalOffset;
    double focalStep = 3.0;
    double minThickness = 0.5;
    double maxThickness = 5.0;
    double thickStep = 0.05;
    double minAngle = 0.0;
    double maxAngle = 70.0;
    double angleStep = 70.0/2500.0;
    double maxIntensity = 1E19;

    // int arraySize = 231 * 5001;
    // double x[arraySize][3] = {};
    double thickness = minThickness;
    double focalDepth = 0.0;

    double thickList[231] = {};
    double focalList[231] = {};

    int counter = 0;
    cout << "Starting sweep up" << endl;
    while (focalDepth<maxFocalOffset) { // initial sweep up
        thickList[counter] = thickness;
        focalList[counter] = focalDepth;
        focalDepth+=focalStep;
        counter++;
    }
     cout << "Starting sweep down" << endl;
    while (focalDepth>minFocalOffset) { // initial sweep down
        thickList[counter] = thickness;
        focalList[counter] = focalDepth;
        focalDepth-=focalStep;
        counter++;
    }
     cout << "Sweeping back to 0" << endl;
    while (focalDepth<0) { // back to initial conditions
        thickList[counter] = thickness;
        focalList[counter] = focalDepth;
        focalDepth+=focalStep;
        counter++;
    }
    cout << "Finishing thickness-focus scan" << endl;
    while (thickness<=maxThickness) {
        thickList[counter] = thickness;
        focalList[counter] = focalDepth;
        thickness+=thickStep;
        focalDepth+=focalStep;
        counter++;
        if(focalDepth==maxFocalOffset) {
            while (focalDepth>minFocalOffset) {
                thickList[counter] = thickness;
                focalList[counter] = focalDepth;
                focalDepth -= focalStep;
                counter++;
            }
        }
    }
    cout << "Scan finished" << endl;

    double currentAngle = minAngle;
    double angleList[5001] = {};
    counter = 0;
    cout << "Sweeping angle up" << endl;
    while (currentAngle <= maxAngle) {
        angleList[counter] = currentAngle;
        currentAngle += angleStep;
        counter++;
    }
    currentAngle -= angleStep;
    cout << "Sweeping angle down" << endl;
    while (currentAngle >= minAngle) {
        angleList[counter] = currentAngle;
        currentAngle-= angleStep;
        counter++;
    }
    int arraySize = 231*5001;
    counter = 0;
    cout << "Starting final array generation" << endl;
    vector<vector<double>> parameterArray(1155231, vector<double> (3, 0));
    int index = 0;
    for (int i = 0; i < 231; i++) {
        // cout << i << endl;
        for (int j = 0; j < 5001; j++) {
            // cout << thickList[i] << endl;
            //cout << focalList[i] << endl;
            //cout << angleList[j] << endl;
            index = (i*5001) + j;
            parameterArray[index][0] = thickList[i];
            parameterArray[index][1] = focalList[i];
            parameterArray[index][2] =  maxIntensity * pow(cos(angleList[j] * M_PI / 180.0), 2.0);
            //cout << parameterArray[counter][2] << endl;
            // if (counter % 100000 == 0) {
            //     cout << "100000 points down" << endl;
            // }
        }
    }

    cout << "Vector generated" << endl;
    // Export vector to CSV
    ofstream myfile ("Campaign1.csv");
    myfile <<  "Thickness,Focus,Intensity" << endl;
    for (int i = 0; i < parameterArray.size(); i++) {
        myfile << parameterArray[i][0] << "," << parameterArray[i][1] << ".0," << parameterArray[i][2] << endl;
    }
    cout << "Table exported" << endl;
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "sec" << std::endl;
    return 0;
}