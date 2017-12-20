#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Threshold to avoid division by zero
#define THRESHOLD 0.001



UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = false;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;


    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.4; // best values somewhere between 0 and 1

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.4; // best values somewhere between 0 and 1

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

    // State dimensions hard-coded
    n_z_ = 3;
    n_x_ = 5;
    n_aug_ = 7;

    H_lidar_ = MatrixXd(2, 4);
    H_lidar_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    R_radar_ = MatrixXd(n_z_, n_z_);
    R_radar_ << std_radr_*std_radr_, 0, 0,
                0, std_radphi_*std_radphi_, 0,
                0, 0, std_radrd_*std_radrd_;

    R_lidar_ = MatrixXd(2, 2);
    R_lidar_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_;

    MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
    Xsig_pred_.fill(0.0);



    lambda_ = 3-n_aug_;

    weights_ = VectorXd(2*n_aug_+1);

    //set weights
    weights_(0) = lambda_/(lambda_+n_aug_); 

    for (int i=1; i<2*n_aug_+1; ++i)
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
        double px, py; 
        double vx, vy, v;
        double psi, psi_dot;
        double phi, rho, rho_dot;

        P_ = MatrixXd::Identity(n_x_, n_x_);

        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            px = meas_package.raw_measurements_(0);
            py = meas_package.raw_measurements_(1);

            // [px, py, v, yaw, yaw_dot]
            x_(0) = px;
            x_(1) = py;
            x_(2) = 0;
            x_(3) = 0;
            x_(4) = 0; 
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            rho = meas_package.raw_measurements_(0);
            phi = meas_package.raw_measurements_(1);
            rho_dot = meas_package.raw_measurements_(2);

            px = rho * cos(phi);
            py = rho * sin(phi);


            // [px, py, v, yaw, yaw_dot]
            x_(0) = px;
            x_(1) = py;
            x_(2) = 0;
            x_(3) = 0;
            x_(4) = 0;
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    

    double dt; // Time in seconds
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
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

    // Create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0.0f;
    x_aug(n_x_+1) = 0.0f;

    // Create augmented covariance matrix
    MatrixXd Q = MatrixXd(2, 2);

    Q << std_a_*std_a_, 0, 
         0, std_yawdd_*std_yawdd_;


    P_aug.topLeftCorner(n_x_, n_x_) = P_;     
    P_aug.bottomRightCorner(2, 2) = Q;


    // Generate augmented sigma points
    MatrixXd A_aug = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    for (int i=0; i < n_aug_; ++i) 
    {
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * A_aug.col(i);
        Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_+n_aug_) * A_aug.col(i);    
    }

    //cout << A_aug << endl;

    // Reinitialize with zero
    Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1);

    x_.fill(0.0);
    P_.fill(0.0);

    // Predict sigma points  
    for (int i=0; i < 2*n_aug_+1; ++i)
    {
        const double px = Xsig_aug.col(i)(0);
        const double py = Xsig_aug.col(i)(1);
        const double v = Xsig_aug.col(i)(2);
        const double yaw = Xsig_aug.col(i)(3);
        const double yaw_dot = Xsig_aug.col(i)(4);
        const double nu_a = Xsig_aug.col(i)(5);
        const double nu_psidd = Xsig_aug.col(i)(6);

        //cout << px << "\t" << py << endl;

        // Don't forget to exclude division by zero!    
        if (fabs(yaw_dot) > THRESHOLD)
        {
            Xsig_pred_.col(i)(0) = px + v/yaw_dot * (sin(yaw+yaw_dot*delta_t)-sin(yaw)) + 0.5f * delta_t*delta_t * cos(yaw) * nu_a;
            Xsig_pred_.col(i)(1) = py + v/yaw_dot * (-cos(yaw+yaw_dot*delta_t)+cos(yaw)) + 0.5f * delta_t*delta_t * sin(yaw) * nu_a;
            Xsig_pred_.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred_.col(i)(3) = yaw + yaw_dot * delta_t + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred_.col(i)(4) = yaw_dot + delta_t * nu_psidd;
        }
        else
        {
            Xsig_pred_.col(i)(0) = px + v * cos(yaw) * delta_t + 0.5f * delta_t*delta_t * cos(yaw) * nu_a;
            Xsig_pred_.col(i)(1) = py + v * sin(yaw) * delta_t + 0.5f * delta_t*delta_t * sin(yaw) * nu_a;
            Xsig_pred_.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred_.col(i)(3) = yaw + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred_.col(i)(4) = delta_t * nu_psidd; // psi_dot is not necessary
        }


        // Predicted mean and covariance
        x_ += weights_(i) * Xsig_pred_.col(i);

        VectorXd delta_x = Xsig_pred_.col(i)-x_;
        delta_x(3) = normalization(delta_x(3));

        P_ += weights_(i) * delta_x * delta_x.transpose(); 
    }
    //cout << Xsig_pred_.col(2)(0) << "\t" << Xsig_pred_.col(2)(1) << endl;
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
    MatrixXd H = MatrixXd(n_z_, n_x_);
    VectorXd y = VectorXd(n_z_);
    VectorXd z = VectorXd(n_z_);
    VectorXd z_pred = VectorXd(n_z_);
    MatrixXd S, K;
    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

    MatrixXd PHt;
    PHt = P_ * H_lidar_.transpose();

    for (int i=0; i<n_z_; ++i)
        z(i) = meas_package.raw_measurements_(i);


    y = z - H * x_;
    S = H * PHt + R_lidar_;

    // Kalman gain
    K = PHt * S.inverse();

    // new state
    x_ = x_ + K * y;
    P_ = (I - K * H) * P_;


    /**
     *  Normalized Innovation Squared (NIS)
     */
    /*VectorXd nis = VectorXd(n_z_);
    nis = (z-z_pred).transpose() * S.inverse() * (z-z_pred);
    ofstream lidar_NIS_handle("./lidar_NIS.dat", ios::out | ios::app);
    lidar_NIS_handle << nis << endl;
    lidar_NIS_handle.close();*/
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
    
    float px, py;
    float v;
    float yaw;
    float vx, vy;
    float rho, phi, rho_dot;

    MatrixXd Zsig = MatrixXd(n_z_, 2*n_aug_+1);
    Zsig.fill(0.0);
    VectorXd z = VectorXd(3);
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);


    //calculate measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_, n_z_);
    MatrixXd Tc = MatrixXd(n_x_, n_z_);
    MatrixXd K;

    S.fill(0.0);
    Tc.fill(0.0);


    // Measurement values
    for (int i=0; i<n_z_; ++i) {
        z(i) = meas_package.raw_measurements_(i);
    }


    // Measurement sigma point prediction
    for (int i=0; i<2*n_aug_+1 ; ++i)
    {
        px = Xsig_pred_(0,i);
        py = Xsig_pred_(1,i);
        v = Xsig_pred_(2,i);
        yaw = Xsig_pred_(3,i);
        
        //cout << px << "\t" << py << endl;

        vx = v*cos(yaw);
        vy = v*sin(yaw);

        rho = sqrt(px*px+py*py);
        if (rho > THRESHOLD) {
            phi = atan2(py, px);    // returns values between -PI and +PI
            rho_dot = (px*vx+py*vy)/rho;
        }
        else {
            phi = 0.0;
            rho_dot = 0.0;
        }
        
        Zsig.col(i) << rho, phi, rho_dot;

        // Calculate mean predicted measurement
        z_pred += weights_(i) * Zsig.col(i); 
    }



    for (int i=0; i<2*n_aug_+1 ; ++i) {
        // Predicted measurement covariance
        VectorXd delta_z = Zsig.col(i)-z_pred;

        delta_z(1) = normalization(delta_z(1));
        
        S += weights_(i) * delta_z * delta_z.transpose();

        // Calculate cross correlation matrix
        VectorXd delta_x = Xsig_pred_.col(i)-x_;

        delta_x(3) = normalization(delta_x(3));
        
        Tc += weights_(i) * delta_x * delta_z.transpose();  
    }


    

    S = S + R_radar_; // Add measurement error matrix
    
    K = Tc * S.inverse(); // Calculate Kalman gain K;


    // Update state mean and covariance matrix
    x_ = x_ + K * (z-z_pred);
    P_ = P_ - K * S * K.transpose();

    //cout << x_(2) << "\t" << x_(3) << endl;
    /**
     *  Normalized Innovation Squared (NIS)
     */
    /*VectorXd nis = VectorXd(n_z_);
    nis = (z-z_pred).transpose() * S.inverse() * (z-z_pred);
    ofstream lidar_NIS_handle("./radar_NIS.dat", ios::out | ios::app);
    lidar_NIS_handle << nis << endl;
    lidar_NIS_handle.close();*/

}
