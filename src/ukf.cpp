#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3; //30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // TODO: Complete the initialization. See ukf.h for other member properties.
    // Hint: one or more values initialized above might be wildly off...
    lambda_ = 3.0f;

    MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

    // State dimensions hard-coded
    n_x_ = 5;
    n_aug_ = 7;

    weights_ = VectorXd(2*n_aug_+1);

    //set weights
    weights_(0) = lambda_/(lambda_+n_aug_); 
    for (int i=1; i<2*n_aug_+1; i++)
        weights_(i) = 0.5f/(lambda_+n_aug_);

    is_initialized_ = false;
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{   
    // TODO: Complete this function! Make sure you switch between lidar and radar measurements.

    // Initialization
    if (!is_initialized_)
    {
        float px, py, v, psi, psi_dot;
        float ro, phi, ro_dot;

        x_ << 0.1f, 0.1f, 0.1f, 0.1f, 0.1f;
        P_ = MatrixXd::Identity(n_x_, n_x_);

        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            px = meas_package.raw_measurements_(0);
            py = meas_package.raw_measurements_(1);

            x_(0) = px;
            x_(1) = py; 
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            ro = meas_package.raw_measurements_(0);
            phi = meas_package.raw_measurements_(1);

            px = ro * cos(phi);
            py = ro * sin(phi);

            x_(0) = px;
            x_(1) = py;
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    

    float dt; // Time in seconds
    dt = (meas_package.timestamp_ - time_us_) / 1000000.f;
    time_us_ = meas_package.timestamp_;


    Prediction(dt);


    // Update
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == true)
    {
        UpdateLidar(meas_package);    
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == true)
    {
        UpdateRadar(meas_package);
    }


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
    // TODO: Complete this function! Estimate the object's location. Modify the state vector, x_. Predict sigma points, the state, and the state covariance matrix.
    float px, py, v, psi, psi_dot, nu_a, nu_psidd;
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
        
    // Create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0.0f;
    x_aug(n_x_+1) = 0.0f;

    // Create augmented covariance matrix
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    
    MatrixXd Q = MatrixXd(2, 2);
    Q << std_a_*std_a_, 0, 
         0, std_yawdd_*std_yawdd_;
    P_aug.bottomRightCorner(2, 2) = Q;



    // Generate augmented sigma points
    Xsig_aug.col(0) = x_aug;
    MatrixXd A_aug = P_aug.llt().matrixL();
    for (int i=0; i < n_aug_; i++) 
    {
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*A_aug.col(i);
        Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_+n_aug_)*A_aug.col(i);    
    }


    // Predict sigma points  
    for (int i=0; i < 2*n_aug_+1; i++)
    {
        px = Xsig_aug.col(i)(0);
        py = Xsig_aug.col(i)(1);
        v = Xsig_aug.col(i)(2);
        psi = Xsig_aug.col(i)(3);
        psi_dot = Xsig_aug.col(i)(4);
        nu_a = Xsig_aug.col(i)(5);
        nu_psidd = Xsig_aug.col(i)(6);

        Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1); // That seemed to be necessary to solve the bug!!! (ML, Nov, 11th)

        // Don't forget to exclude division by zero!    
        if (psi_dot != 0.0f)
        {
            Xsig_pred_.col(i)(0) = px + v/psi_dot * (sin(psi+psi_dot*delta_t)-sin(psi)) + 0.5f * delta_t*delta_t * cos(psi) * nu_a;
            Xsig_pred_.col(i)(1) = py + v/psi_dot * (-cos(psi+psi_dot*delta_t)+cos(psi)) + 0.5f * delta_t*delta_t * sin(psi) * nu_a;
            Xsig_pred_.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred_.col(i)(3) = psi + psi_dot * delta_t + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred_.col(i)(4) = psi_dot + delta_t * nu_psidd;
        }
        else if (psi_dot == 0.0f)
        {
            Xsig_pred_.col(i)(0) = px + v * cos(psi) * delta_t + 0.5f * delta_t*delta_t * cos(psi) * nu_a;
            Xsig_pred_.col(i)(1) = py + v * sin(psi) * delta_t + 0.5f * delta_t*delta_t * sin(psi) * nu_a;
            Xsig_pred_.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred_.col(i)(3) = psi + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred_.col(i)(4) = psi_dot + delta_t * nu_psidd; // psi_dot not necessary
        }
    }



    // Predicted mean and covariance
    // temporary variables
    VectorXd x_mean = VectorXd::Zero(n_x_);
    MatrixXd P_mean = MatrixXd::Zero(n_x_, n_x_);

    //predict state mean
    for (int i=0; i<2*n_aug_+1; i++)
        x_mean += weights_(i) * Xsig_pred_.col(i);

    x_ = x_mean;

    //predict state covariance matrix
    for (int i=0; i<2*n_aug_+1; i++)
        P_mean += weights_(i) * (Xsig_pred_.col(i)-x_) * (Xsig_pred_.col(i)-x_).transpose();

    P_ = P_mean;
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
    // TODO: Complete this function! Use lidar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_.
    //You'll also need to calculate the lidar NIS.

    /*
     *  We are allowed to use the linear Kalman equations here
     */
    // Initialization the measurement matrix for the LIDAR
    int n_z = meas_package.raw_measurements_.size();
    MatrixXd H = MatrixXd(n_z, n_x_);
    VectorXd y = VectorXd(n_z);
    VectorXd z = VectorXd(n_z);
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S, K;
    MatrixXd R = MatrixXd(n_z, n_z);
    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

    H << 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0;

    R << std_laspx_ * std_laspx_, 0,
         0, std_laspy_ * std_laspy_;

    for (int i=0; i<2; i++)
        z(i) = meas_package.raw_measurements_(i);

    y = z - H * x_;
    S = H * P_ * H.transpose() + R;

    //Kalman gain
    K = P_ * H.transpose() * S.inverse();

    // new state
    x_ = x_ + K * y;
    P_ = (I - K * H) * P_;


    /**
     *  Normalized Innovation Squared (NIS)
     */
    VectorXd nis = VectorXd(n_z);
    nis = (z-z_pred).transpose() * S.inverse() * (z-z_pred);
    ofstream lidar_NIS_handle("./lidar_NIS.dat", ios::out | ios::app);
    lidar_NIS_handle << nis << endl;
    lidar_NIS_handle.close();
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
    // TODO: Complete this function! Use radar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_.
    // You'll also need to calculate the radar NIS.
    //transform sigma points into measurement space
    
    float px, py, v;
    float ro, phi, ro_dot;
    int n_z = meas_package.raw_measurements_.size();
    MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
    VectorXd z = VectorXd(3);
    VectorXd z_pred = VectorXd::Zero(3);

    // measurement values
    for (int i=0; i<3; i++)
        z(i) = meas_package.raw_measurements_(i);

    for (int i=0; i<2*n_aug_+1 ; i++)
    {
        px = Xsig_pred_(0,i);
        py = Xsig_pred_(1,i);
        v = Xsig_pred_(2,i);

        // Check for division by zero!
        ro = sqrt(px*px+py*py);
        phi = atan2(py, px);    // returns values between -PI and +PI

        if (ro >= 0.1f)
            ro_dot = (px*v+py*v) / ro;
        else
            ro_dot = 1.0f;

        Zsig.col(i) << ro, phi, ro_dot;
    }

    //calculate mean predicted measurement
    for (int i=0; i<2*n_aug_+1; i++)
        z_pred += weights_(i) * Zsig.col(i); 

    //calculate measurement covariance matrix S
    MatrixXd R = MatrixXd(n_z, n_z);
    MatrixXd S = MatrixXd::Zero(n_z, n_z);

    R << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;

    // predicted measurement covariance
    for (int i=0; i<2*n_aug_+1; i++)
        S += weights_(i) * (Zsig.col(i)-z_pred) * (Zsig.col(i)-z_pred).transpose();

    // add measurement error matrix
    S += R;

    //calculate cross correlation matrix
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    
    for (int i=0; i < 2*n_aug_+1; i++)
    {
        Tc += weights_(i) * (Xsig_pred_.col(i)-x_) * (Zsig.col(i)-z_pred).transpose();  
    }

    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();


    //update state mean and covariance matrix
    x_ = x_ + K * (z-z_pred);
    P_ = P_ - K * S * K.transpose();


    /**
     *  Normalized Innovation Squared (NIS)
     */
    VectorXd nis = VectorXd(n_z);
    nis = (z-z_pred).transpose() * S.inverse() * (z-z_pred);
    ofstream lidar_NIS_handle("./radar_NIS.dat", ios::out | ios::app);
    lidar_NIS_handle << nis << endl;
    lidar_NIS_handle.close();
}
