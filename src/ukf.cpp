#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

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
    std_a_ = 30;

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
    lambda_ = 3.f;
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
        P_ = MatrixXd::Identity(x_.size(), x_.size());

        if (meas_package.sensor_type_ == MeasurementPackage::LIDAR)
        {
            px = meas_package.raw_measurements_(0);
            py = meas_package.raw_measurements_(1);

            x_(0) = px;
            x_(1) = py; 
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LIDAR)
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
    dt = (meas_package.timestamp_ - time_us_) / 1000000.f 
    time_us_ = meas_package.timestamp_;



    // Prediction
    // Generate sigma points (augmented points)
    MatrixXd A = P_.llt().matrixL();
    
    Xsig_pred_.col(0) = x_;
    for (int i=0; i < n_; i++)
    {
        Xsig_pred_.col(i+1) = x_ + sqrt(lambda+n_)*A.col(i);
        Xsig_pred_.col(i+n_+1) = x_ - sqrt(lambda+n_)*A.col(i);    
    }


    // Predict sigma points
    Prediction(dt);
    

    // Update

    
    if (meas_package.sensor_type_ == MeasurementPackage::LIDAR && use_laser_==true)
    {
        UpdateLidar(meas_package);    
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_==true)
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
  
  
    for (int i=0; i < 2*n_aug+1; i++)
    {
        px = Xsig_aug.col(i)(0);
        py = Xsig_aug.col(i)(1);
        v = Xsig_aug.col(i)(2);
        psi = Xsig_aug.col(i)(3);
        psi_dot = Xsig_aug.col(i)(4);
        nu_a = Xsig_aug.col(i)(5);
        nu_psidd = Xsig_aug.col(i)(6);


        // Don't forget to exclude division by zero!    
        if (psi_dot != 0)
        {
            Xsig_pred.col(i)(0) = px + v/psi_dot * (sin(psi+psi_dot*delta_t)-sin(psi)) + 0.5f * delta_t*delta_t * cos(psi) * nu_a;
            Xsig_pred.col(i)(1) = py + v/psi_dot * (-cos(psi+psi_dot*delta_t)+cos(psi)) + 0.5f * delta_t*delta_t * sin(psi) * nu_a;
            Xsig_pred.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred.col(i)(3) = psi + psi_dot * delta_t + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred.col(i)(4) = psi_dot + delta_t * nu_psidd;
        }
        else if (psi_dot == 0)
        {
            Xsig_pred.col(i)(0) = px + v * cos(psi) * delta_t + 0.5f * delta_t*delta_t * cos(psi) * nu_a;
            Xsig_pred.col(i)(1) = py + v * sin(psi) * delta_t + 0.5f * delta_t*delta_t * sin(psi) * nu_a;
            Xsig_pred.col(i)(2) = v + delta_t * nu_a;
            Xsig_pred.col(i)(3) = psi + 0.5f * delta_t*delta_t * nu_psidd;
            Xsig_pred.col(i)(4) = psi_dot + delta_t * nu_psidd;
        }
    }

    /**
     * Predicted mean and covariance
     */

    //set weights
    weights(0) = lambda/(lambda+n_aug); 
    for (int i=1; i<2*n_aug+1; i++)
        weights(i) = 0.5f/(lambda+n_aug); 
  
    //predict state mean
    x += weights(0) * Xsig_pred.col(0);
    for (int i=1; i<2*n_aug+1; i++)
        x += weights(i) * Xsig_pred.col(i);
  
    //predict state covariance matrix
    for (int i=0; i<2*n_aug+1; i++)
        P += weights(i) * (Xsig_pred.col(i)-x) * (Xsig_pred.col(i)-x).transpose();
  

}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
    // TODO: Complete this function! Use lidar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_.
    //You'll also need to calculate the lidar NIS.

    // Initializon the measurement matrix for the LIDAR
    MatrixXd H = MatrixXd(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;


    VectorXd y = VectorXd(2);

    MatrixXd S, K, Ht, Si;
    y = z - H_ * x_;
    Ht = H_.transpose();
    S = H_ * P_ * Ht + R_;
    Si = S.inverse();

    /**
     *  Calculating Kalman gain and update (x, P)
     */

    K = P_ * Ht * Si;
    // new state
    x_ = x_ + K * y;
    MatrixXd I = MatrixXd::Identity(4, 4);
    P_ = (I - K * H_) * P_;


    /**
     *  NIS
     */

    (z-z_pred).transpose() * S.inverse() * (z-z_pred)


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
    float px, py, v, psi;
    float ro, phi, ro_dot;

    for (int i=0; i<2*n_aug+1 ; i++)
    {
        px = Xsig_pred(0,i);
        py = Xsig_pred(1,i);
        v = Xsig_pred(2,i);
        psi = Xsig_pred(3,i);

        ro = sqrt(px*px+py*py);
        phi = atan2(py, px);
        ro_dot = (px*cos(psi)*v + py*sin(psi)*v)/ro;
        Zsig.col(i) << ro, phi, ro_dot;
    }

    //std::cout << Zsig << std::endl;

    //calculate mean predicted measurement
    for (int i=0; i<2*n_aug+1; i++)
        z_pred += weights(i) * Zsig.col(i); 

    //calculate measurement covariance matrix S
    MatrixXd R = MatrixXd(n_z, n_z);

    R << std_radr*std_radr, 0, 0,
        0, std_radphi*std_radphi, 0,
        0, 0, std_radrd*std_radrd;


    for (int i=0; i<n_aug; i++)
        S += weights(i) * (Zsig.col(i)-z_pred) * (Zsig.col(i)-z_pred).transpose();

    S += R;


    /**
     *  Calculating Kalman gain and finally update (x, P)
     */

    //calculate cross correlation matrix
    for (int i=0; i < 2*n_aug+1; i++)
    {
    Tc += weights(i) * (Xsig_pred.col(i)-x) * (Zsig.col(i)-z_pred).transpose();  
    }

    //std::cout << Tc << std::endl;

    //calculate Kalman gain K;
    MatrixXd K = MatrixXd(n_x, n_z);

    K = Tc * S.inverse();
    //std::cout << K << std::endl;

    //update state mean and covariance matrix
    x = x + K * (z-z_pred);
    P = P - K * S * K.transpose();


    /**
     *  NIS
     */

}
