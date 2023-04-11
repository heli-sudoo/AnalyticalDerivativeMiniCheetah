#include <iostream>
#include <vector>
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

using namespace pinocchio;
using namespace Eigen;


class UnderActuatedModel : public Model
{
public:
  UnderActuatedModel () {
    S.setZero(18, 12);
    S.bottomRows(12).setIdentity();
    nu = 12;
  }

public:
  MatrixXd S;
  size_t nu;
};


template<typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, 
         typename LagrangeType>
void contactDynamics(const UnderActuatedModel & model, Data & data,                        
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,
                         const Eigen::MatrixBase<TangentVectorType2> & tau_12,
                         Eigen::MatrixBase<TangentVectorType2> & qdd,
                         const std::vector<FrameIndex>& contactFrameIds,
                         Eigen::MatrixBase<LagrangeType> & lambda)
{
  size_t n_contacts = contactFrameIds.size();

  computeJointJacobians(model, data, q);

  MatrixXd J(6*n_contacts, model.nv); 
  J.setZero();

  Data::Matrix6x Jc(6, model.nv);
  for (size_t i(0); i < n_contacts; ++i)
  {
    Jc.setZero();
    getFrameJacobian(model, data, contactFrameIds[i], LOCAL_WORLD_ALIGNED, Jc);
    J.block(6*i, 0, 6, model.nv) = Jc;
  }
  
  VectorXd gamma(6 * n_contacts);
  Motion::Vector6 gamma_c;

  forwardKinematics(model, data, q, v, VectorXd::Zero(model.nv));

  for (size_t i(0); i < n_contacts; ++i)
  {    
    gamma_c = getFrameAcceleration(model, data, contactFrameIds[i], LOCAL_WORLD_ALIGNED).toVector();
    gamma.segment(6*i, 6) = gamma_c;
  }
  
  qdd = forwardDynamics(model, data, q, v, model.S*tau_12, J, gamma, 1e-12);
  lambda = data.lambda_c;  
}

template<typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2, 
         typename lambdaType,
         typename qddPartialType1, typename qddPartialType2, typename qddPartialType3,
         typename lambdaPartialType1, typename lambdaPartialType2, typename lambdaPartialType3>
void contactDynamicsDerivatives(const UnderActuatedModel & model, Data & data,                        
                         const Eigen::MatrixBase<ConfigVectorType> & q,
                         const Eigen::MatrixBase<TangentVectorType1> & v,                         
                         const Eigen::MatrixBase<TangentVectorType2> & qdd,                     
                         const std::vector<FrameIndex>& contactFrameIds,     
                         const Eigen::MatrixBase<lambdaType>& lambdasFootFrame,                    
                         Eigen::MatrixBase<qddPartialType1>& dqdd_dtau,
                         Eigen::MatrixBase<qddPartialType2>& dqdd_dq, 
                         Eigen::MatrixBase<qddPartialType3>& dqdd_dv,
                         Eigen::MatrixBase<lambdaPartialType1>& dlambda_dtau,
                         Eigen::MatrixBase<lambdaPartialType2>& dlambda_dq,
                         Eigen::MatrixBase<lambdaPartialType3>& dlambda_dv)
{
  size_t n_contacts = contactFrameIds.size();

  // Compute concatenated contact Jacobian
  computeJointJacobians(model, data, q);

  MatrixXd J(6*n_contacts, model.nv); 
  J.setZero();

  Data::Matrix6x Jc(6, model.nv);
  for (size_t i(0); i < n_contacts; ++i)
  {
    Jc.setZero();
    getFrameJacobian(model, data, contactFrameIds[i], LOCAL_WORLD_ALIGNED, Jc);
    J.block(6*i, 0, 6, model.nv) = Jc;
  }

  // Compute the inverse of the KKT matrix
  MatrixXd Ainv(model.nv + 6*n_contacts, model.nv+6*n_contacts);
  computeKKTContactDynamicMatrixInverse(model, data, q, J, Ainv);

  // Compute dqdd_dtau, dlambda_dtau
  dqdd_dtau = Ainv.topRows(model.nv).leftCols(model.nv) * model.S; // should multiple the selection matrix ? (I think so)
  dlambda_dtau = Ainv.bottomRows(6).leftCols(model.nv) * model.S;  

  // Compute equivalent spatial contact forces acting on the parent joint frame
  // Alternative method to express fext in joint frame https://github.com/leggedrobotics/ocs2/blob/ebde452b10d0eceaac45364f7bb8f0ac1038b637/ocs2_pinocchio/ocs2_centroidal_model/src/CentroidalModelRbdConversions.cpp#L189 
  PINOCCHIO_ALIGNED_STD_VECTOR(Force) fext((size_t)model.njoints, Force::Zero());
  SE3 jointFrameToFootFrame;
  jointFrameToFootFrame.setIdentity();
  for (size_t i = 0; i < n_contacts; i++)
  {
    const auto& lambda = lambdasFootFrame.segment(6*i, 6);
    Force contactForceFootFrame(lambda);
    jointFrameToFootFrame.translation() = model.frames[contactFrameIds[i]].placement.translation();
    Force contactForceJointFrame(jointFrameToFootFrame.actInv(contactForceFootFrame));
    fext[model.frames[contactFrameIds[i]].parent] = contactForceJointFrame;
  }
  
  // Derivatives of inverse dynamics
  computeRNEADerivatives(model, data, q, v, qdd, fext);

  // Compute dqdd_dv, dlambda_dv
  dqdd_dv = -Ainv.topRows(model.nv).leftCols(model.nv) * (model.S*data.dtau_dv.bottomRows(12));
  dlambda_dv = -Ainv.bottomRows(6).leftCols(model.nv) * (model.S*data.dtau_dv.bottomRows(12)); 

  // Compute partial derivative of foot spatial acceleration w.r.t. joint angle, and joint velocity, and joint acceleration
  Data::Matrix6x v_partial_dq(6,model.nv), a_partial_dq(6,model.nv), a_partial_dv(6,model.nv), a_partial_da(6,model.nv);
  computeForwardKinematicsDerivatives(model,data, q, VectorXd::Zero(model.nv), qdd);

  MatrixXd a_partial_dq_all(6 * n_contacts, model.nv);
  a_partial_dq_all.setZero();
  for (size_t i = 0; i < n_contacts; i++)
  {
    v_partial_dq.setZero(); a_partial_dq.setZero(); a_partial_dv.setZero(); a_partial_da.setZero();
    getFrameAccelerationDerivatives(model,data, contactFrameIds[i], 
                                    LOCAL_WORLD_ALIGNED, v_partial_dq,a_partial_dq,a_partial_dv,a_partial_da);

    // Put the derivatives of all contact foot in one giant matrix                                    
    a_partial_dq_all.block(6*i, 0, 6, model.nv) = a_partial_dq;                                  
  }
  
  
  // Get dqdd_dq, dlambda_dq
  // Rohan Thesis (3.16) for analytical derivative of KKT 
  // Inverse of KKT matrix https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#a14be42b6e0582bc3dd9e91094e573349
  dqdd_dq = -Ainv.topRows(model.nv).leftCols(model.nv) * (model.S*data.dtau_dq.bottomRows(12));
  dqdd_dq -= Ainv.topRows(model.nv).rightCols(6 * n_contacts) * a_partial_dq_all;
  
  dlambda_dq = Ainv.bottomRows(6).leftCols(model.nv) * (model.S*data.dtau_dq.bottomRows(12));
  dlambda_dq += Ainv.bottomRows(6).rightCols(6 * n_contacts) * a_partial_dq_all;
}                         

int main()
{  
  /* Create a floating-base MC model with ZYX-euler anglre representation */
  Model fixed_base_model;
  Model float_base_model;
  UnderActuatedModel model;

  const std::string urdf_filename = std::string("../urdf/mini_cheetah_simple.urdf");
  pinocchio::urdf::buildModel(urdf_filename, fixed_base_model);



  
  float_base_model.name = "float_base";
  int jnt_parent_id = 0;
  std::string jnt_name("fb_translation");
  int jnt_id = float_base_model.addJoint(jnt_parent_id, pinocchio::JointModelTranslation(), pinocchio::SE3::Identity(), jnt_name);

  jnt_parent_id = jnt_id;
  jnt_name = std::string("fb_euler");
  // Note that the angular velocity would be in Z-Y-X order
  jnt_id = float_base_model.addJoint(jnt_parent_id, pinocchio::JointModelSphericalZYX(), pinocchio::SE3::Identity(), jnt_name);

  int frame_id = float_base_model.addFrame(Frame("float_base", 
                                                 jnt_id, 
                                                 float_base_model.getFrameId("universe"),
                                                 SE3::Identity(),
                                                 JOINT));

  pinocchio::appendModel(float_base_model, fixed_base_model, frame_id, pinocchio::SE3::Identity(), model);


  // Print model information
  std::cout << "--------------- Model Information -----------------\n";
  std::cout << "Robot name: " << model.name << std::endl;
  std::cout << "Number of joints: " << model.njoints << std::endl;
  std::cout << "Number of bodies: " << model.nbodies << std::endl;
  std::cout << "Number of position variables nq: " << model.nq << std::endl;
  std::cout << "Number of velocity variables nv: " << model.nv << std::endl;
  for (size_t j(0); j < model.njoints; ++j)
  {
    std::cout << "Joint " << j << " name: " << model.names[j] << std::endl;
  }

  for (const auto& frame : model.frames)
  {
    std::cout << "Frame id: " << model.getFrameId(frame.name) << " Frame name: " << frame.name << std::endl;
  }
  

  
  // Create model-associated data structure
  Data data(model);

  // Randomize configuration, velocity, and toruqe
  srand((unsigned int) time(0));
  VectorXd q = VectorXd::Zero(model.nq);
  VectorXd v = VectorXd::Ones(model.nv);
  VectorXd tau = VectorXd::Random(model.nu);

  // Shcedule contact
  std::vector<std::string> contactFrameNames;  
  std::vector<FrameIndex> contactFrameIds;
  contactFrameNames.push_back("foot_fr");
  size_t N_contacts = contactFrameNames.size();
  for (const auto& name : contactFrameNames)
  {
    contactFrameIds.push_back(model.getFrameId(name));
  }

  // Compute forward KKT dynamics using CRBA 
  VectorXd qdd(model.nv), lambda(6 * N_contacts);  
  contactDynamics(model, data, q, v, tau, qdd, contactFrameIds, lambda);

  // Compute partial derivatives of the forward KKT dynamics
  MatrixXd dqdd_dtau(model.nv, model.nu); dqdd_dtau.setZero();
  MatrixXd dqdd_dv(model.nv, model.nv); dqdd_dv.setZero();
  MatrixXd dqdd_dq(model.nv, model.nv); dqdd_dq.setZero();

  MatrixXd dlambda_dtau(6*N_contacts, model.nu); dlambda_dtau.setZero();
  MatrixXd dlambda_dq(6*N_contacts, model.nv); dlambda_dq.setZero();
  MatrixXd dlambda_dv(6*N_contacts, model.nv); dlambda_dv.setZero();

  contactDynamicsDerivatives(model, data, q, v, qdd, contactFrameIds, lambda,
                             dqdd_dtau, dqdd_dq, dqdd_dv,
                             dlambda_dtau, dlambda_dq, dlambda_dv);                        
}


