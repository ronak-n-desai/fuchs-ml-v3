// Date: 2024-01-03
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
    cout << "starting";
    
    // variable declaration
    const int maxFocalOffset = 30;
    const int minFocalOffset = -maxFocalOffset;
    const double focalStep = 1.0;
    const double minThickness = 0.5;
    const double maxThickness = 5.0;
    const double thickStep = 0.05;
    const double minAngle = 0.0;
    const double maxAngle = 70.0;
    const double angleStep = 0.7;
    const double maxIntensity = 1E19;
    const double maxContrast = 1E-7 / pow(cos(70 * M_PI / 180.0), 2.0);
    const int minMainT = 0, maxMainT = 70, nMain = 700;
    const int minPreT = 0, maxPreT = 70, nPre = 700;
    const int numBounces = 20;
    // declaring lists
    std::vector<double> thickList, preAngleList, mainAngleList;
    // populating lists
    for (double i = minThickness; i <= maxThickness; i += thickStep) {
        thickList.push_back(i);
    }
    for (int i = 0; i <= nMain * numBounces; i++) {
        int imod = i % nMain;
        int doublemod = i % (2 * nMain);
        double preTheta = (static_cast<double>(nPre) / (nMain * numBounces)) * i * 0.1;
        double mainTheta;
        if (imod == doublemod) {
            mainTheta = (-(maxMainT - minMainT) * static_cast<double>(imod)) / nMain + maxMainT;
        } else {
            mainTheta = ((maxMainT + minMainT) * static_cast<double>(imod)) / nMain + minMainT;
        }
        preAngleList.push_back(preTheta);
        mainAngleList.push_back(mainTheta);
    }
    cout << "Thick:" << thickList.size() << endl;
    cout << "Pre Angle:" << preAngleList.size() << endl;
    cout << "Main Angle:" << mainAngleList.size() << endl;
    cout << "Starting grid generation" << endl;
    vector<vector<double>> parameterArray(1274091, vector<double> (3, 0));
    int index = 0;
    for (int i = 0; i < thickList.size(); i++) {
        for (int j = 0; j < mainAngleList.size(); j++) { 
            parameterArray[index][0] = thickList[i];
            parameterArray[index][1] = maxContrast * pow(cos(preAngleList[j] * M_PI / 180.0), 2.0);
            parameterArray[index][2] =  maxIntensity * pow(cos(mainAngleList[j] * M_PI / 180.0), 2.0);
            index++;
        }
    }

    cout << "Vector generated" << endl;
    // Export vector to CSV
    ofstream myfile ("Campaign2.csv");
    myfile <<  "Thickness,Contrast,Intensity" << endl;
    for (int i = 0; i < parameterArray.size(); i++) {
        myfile << parameterArray[i][0] << "," << parameterArray[i][1] << "," << parameterArray[i][2] << endl;
    }

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "sec" << std::endl;
    return 0;
}
