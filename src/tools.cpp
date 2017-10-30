#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth) 
{
	// TODO: Calculate the RMSE here.
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // Check
    if ((estimations.size() == 0) || (estimations.size() != ground_truth.size()))
        return rmse;

    VectorXd resDiff(4);
    
    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        resDiff = estimations[i]-ground_truth[i];
        rmse.array() = rmse.array() + resDiff.array()*resDiff.array();
    }

    rmse = rmse.array()/estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}
