#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include <math.h>

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline double normalization(double angle)
{   
    //std::cout << angle << std::endl; 
    while (angle > M_PI) angle -= 2*M_PI;
    while (angle < -M_PI) angle += 2*M_PI;
    return angle;
}


class UKF 
{  
    public:
        ///* initially set to false, set to true in first call of ProcessMeasurement
        bool is_initialized_;

        ///* if this is false, laser measurements will be ignored (except for init)
        bool use_laser_;

        ///* if this is false, radar measurements will be ignored (except for init)
        bool use_radar_;

        ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
        VectorXd x_;

        ///* state covariance matrix
        MatrixXd P_;

        ///* predicted sigma points matrix
        MatrixXd Xsig_pred_;

        ///* time when the state is true, in us
        long long time_us_;

        ///* Process noise standard deviation longitudinal acceleration in m/s^2
        double std_a_;

        ///* Process noise standard deviation yaw acceleration in rad/s^2
        double std_yawdd_;

        ///* Laser measurement noise standard deviation position1 in m
        double std_laspx_;

        ///* Laser measurement noise standard deviation position2 in m
        double std_laspy_;

        ///* Radar measurement noise standard deviation radius in m
        double std_radr_;

        ///* Radar measurement noise standard deviation angle in rad
        double std_radphi_;

        ///* Radar measurement noise standard deviation radius change in m/s
        double std_radrd_ ;

        MatrixXd R_radar_;
        MatrixXd R_lidar_;
        MatrixXd H_lidar_;

        ///* Weights of sigma points
        VectorXd weights_m_; // weights for the mean values
        VectorXd weights_c_; // weights for the covariance matrices

        ///* State dimensions
        int n_z_radar_;
        int n_z_lidar_;
        int n_x_;

        ///* Augmented state dimension
        int n_aug_;

        ///* van-der-Merwe-coefficients 
        double alpha_;
        double beta_;
        double kappa_;

        ///* Sigma point spreading parameter
        double lambda_;




        /**
         * Constructor
         */
        UKF();

        /**
         * Destructor
         */
        virtual ~UKF();

        /**
         * ProcessMeasurement
         * @param meas_package The latest measurement data of either radar or laser
         */
        void ProcessMeasurement(MeasurementPackage meas_package);


        // Overloaded method:
        VectorXd BicycleModel(VectorXd state, double nu_a, double nu_psidd);
        VectorXd BicycleModel(VectorXd state, double nu_a, double nu_psidd, double dt);

        /**
         * Prediction Predicts sigma points, the state, and the state covariance
         * matrix
         * @param delta_t Time between k and k+1 in s
         */
        void Prediction(double delta_t);



        /**
         * Updates the state and the state covariance matrix using a laser measurement
         * @param meas_package The measurement at k+1
         */
        void UpdateLidar(MeasurementPackage meas_package);

        /**
         * Updates the state and the state covariance matrix using a radar measurement
         * @param meas_package The measurement at k+1
         */
        void UpdateRadar(MeasurementPackage meas_package);
};


#endif /* UKF_H */
