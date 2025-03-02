// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel
#include "boost/date_time.hpp"
#include "boost/shared_array.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4265)
#pragma warning(disable : 4458)
#pragma warning(disable : 4100)
#pragma warning(disable : 4127)
#pragma warning(disable : 4267)
#pragma warning(disable : 4068)
#pragma warning(disable : 4245)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif //_MSC_VER
#include "ros/ros.h"
#include "uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel.h"
#include "visibility_control.h"
#include "MATLABROSMsgInterface.hpp"
#include "ROSPubSubTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //added_mass
        const matlab::data::TypedArray<double> added_mass_arr = arr["AddedMass"];
        size_t nelem = added_mass_arr.getNumberOfElements();
        	msg->added_mass.resize(nelem);
        	std::copy(added_mass_arr.begin(), added_mass_arr.begin()+nelem, msg->added_mass.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'AddedMass' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'AddedMass' is wrong type; expected a double.");
    }
    try {
        //linear_damping
        const matlab::data::TypedArray<double> linear_damping_arr = arr["LinearDamping"];
        size_t nelem = linear_damping_arr.getNumberOfElements();
        	msg->linear_damping.resize(nelem);
        	std::copy(linear_damping_arr.begin(), linear_damping_arr.begin()+nelem, msg->linear_damping.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'LinearDamping' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'LinearDamping' is wrong type; expected a double.");
    }
    try {
        //linear_damping_forward_speed
        const matlab::data::TypedArray<double> linear_damping_forward_speed_arr = arr["LinearDampingForwardSpeed"];
        size_t nelem = linear_damping_forward_speed_arr.getNumberOfElements();
        	msg->linear_damping_forward_speed.resize(nelem);
        	std::copy(linear_damping_forward_speed_arr.begin(), linear_damping_forward_speed_arr.begin()+nelem, msg->linear_damping_forward_speed.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'LinearDampingForwardSpeed' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'LinearDampingForwardSpeed' is wrong type; expected a double.");
    }
    try {
        //quadratic_damping
        const matlab::data::TypedArray<double> quadratic_damping_arr = arr["QuadraticDamping"];
        size_t nelem = quadratic_damping_arr.getNumberOfElements();
        	msg->quadratic_damping.resize(nelem);
        	std::copy(quadratic_damping_arr.begin(), quadratic_damping_arr.begin()+nelem, msg->quadratic_damping.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'QuadraticDamping' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'QuadraticDamping' is wrong type; expected a double.");
    }
    try {
        //volume
        const matlab::data::TypedArray<double> volume_arr = arr["Volume"];
        msg->volume = volume_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Volume' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Volume' is wrong type; expected a double.");
    }
    try {
        //bbox_height
        const matlab::data::TypedArray<double> bbox_height_arr = arr["BboxHeight"];
        msg->bbox_height = bbox_height_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'BboxHeight' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'BboxHeight' is wrong type; expected a double.");
    }
    try {
        //bbox_length
        const matlab::data::TypedArray<double> bbox_length_arr = arr["BboxLength"];
        msg->bbox_length = bbox_length_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'BboxLength' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'BboxLength' is wrong type; expected a double.");
    }
    try {
        //bbox_width
        const matlab::data::TypedArray<double> bbox_width_arr = arr["BboxWidth"];
        msg->bbox_width = bbox_width_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'BboxWidth' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'BboxWidth' is wrong type; expected a double.");
    }
    try {
        //fluid_density
        const matlab::data::TypedArray<double> fluid_density_arr = arr["FluidDensity"];
        msg->fluid_density = fluid_density_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'FluidDensity' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'FluidDensity' is wrong type; expected a double.");
    }
    try {
        //cob
        const matlab::data::StructArray cob_arr = arr["Cob"];
        auto msgClassPtr_cob = getCommonObject<geometry_msgs::Vector3>("geometry_msgs_msg_Vector3_common",loader);
        msgClassPtr_cob->copy_from_struct(&msg->cob,cob_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Cob' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Cob' is wrong type; expected a struct.");
    }
    try {
        //neutrally_buoyant
        const matlab::data::TypedArray<bool> neutrally_buoyant_arr = arr["NeutrallyBuoyant"];
        msg->neutrally_buoyant = neutrally_buoyant_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'NeutrallyBuoyant' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'NeutrallyBuoyant' is wrong type; expected a logical.");
    }
    try {
        //inertia
        const matlab::data::StructArray inertia_arr = arr["Inertia"];
        auto msgClassPtr_inertia = getCommonObject<geometry_msgs::Inertia>("geometry_msgs_msg_Inertia_common",loader);
        msgClassPtr_inertia->copy_from_struct(&msg->inertia,inertia_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Inertia' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Inertia' is wrong type; expected a struct.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","AddedMass","LinearDamping","LinearDampingForwardSpeed","QuadraticDamping","Volume","BboxHeight","BboxLength","BboxWidth","FluidDensity","Cob","NeutrallyBuoyant","Inertia"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/UnderwaterObjectModel");
    // added_mass
    auto currentElement_added_mass = (msg + ctr)->added_mass;
    outArray[ctr]["AddedMass"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel::_added_mass_type::const_iterator, double>({currentElement_added_mass.size(),1}, currentElement_added_mass.begin(), currentElement_added_mass.end());
    // linear_damping
    auto currentElement_linear_damping = (msg + ctr)->linear_damping;
    outArray[ctr]["LinearDamping"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel::_linear_damping_type::const_iterator, double>({currentElement_linear_damping.size(),1}, currentElement_linear_damping.begin(), currentElement_linear_damping.end());
    // linear_damping_forward_speed
    auto currentElement_linear_damping_forward_speed = (msg + ctr)->linear_damping_forward_speed;
    outArray[ctr]["LinearDampingForwardSpeed"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel::_linear_damping_forward_speed_type::const_iterator, double>({currentElement_linear_damping_forward_speed.size(),1}, currentElement_linear_damping_forward_speed.begin(), currentElement_linear_damping_forward_speed.end());
    // quadratic_damping
    auto currentElement_quadratic_damping = (msg + ctr)->quadratic_damping;
    outArray[ctr]["QuadraticDamping"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel::_quadratic_damping_type::const_iterator, double>({currentElement_quadratic_damping.size(),1}, currentElement_quadratic_damping.begin(), currentElement_quadratic_damping.end());
    // volume
    auto currentElement_volume = (msg + ctr)->volume;
    outArray[ctr]["Volume"] = factory.createScalar(currentElement_volume);
    // bbox_height
    auto currentElement_bbox_height = (msg + ctr)->bbox_height;
    outArray[ctr]["BboxHeight"] = factory.createScalar(currentElement_bbox_height);
    // bbox_length
    auto currentElement_bbox_length = (msg + ctr)->bbox_length;
    outArray[ctr]["BboxLength"] = factory.createScalar(currentElement_bbox_length);
    // bbox_width
    auto currentElement_bbox_width = (msg + ctr)->bbox_width;
    outArray[ctr]["BboxWidth"] = factory.createScalar(currentElement_bbox_width);
    // fluid_density
    auto currentElement_fluid_density = (msg + ctr)->fluid_density;
    outArray[ctr]["FluidDensity"] = factory.createScalar(currentElement_fluid_density);
    // cob
    auto currentElement_cob = (msg + ctr)->cob;
    auto msgClassPtr_cob = getCommonObject<geometry_msgs::Vector3>("geometry_msgs_msg_Vector3_common",loader);
    outArray[ctr]["Cob"] = msgClassPtr_cob->get_arr(factory, &currentElement_cob, loader);
    // neutrally_buoyant
    auto currentElement_neutrally_buoyant = (msg + ctr)->neutrally_buoyant;
    outArray[ctr]["NeutrallyBuoyant"] = factory.createScalar(static_cast<bool>(currentElement_neutrally_buoyant));
    // inertia
    auto currentElement_inertia = (msg + ctr)->inertia;
    auto msgClassPtr_inertia = getCommonObject<geometry_msgs::Inertia>("geometry_msgs_msg_Inertia_common",loader);
    outArray[ctr]["Inertia"] = msgClassPtr_inertia->get_arr(factory, &currentElement_inertia, loader);
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message::generatePublisherInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel,uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message::generateSubscriberInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel,uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface>
         uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message::generateRosbagWriterInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel,uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common>>();
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_UnderwaterObjectModel_message, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1