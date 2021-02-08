/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// In this example we will run 3D rigid registration
// with Both Mattes, Tsallis, and NormTsallis Metric;
// Receives at parameters:
// " fixedImageFile   movingImageFile  MetricType ('Mattes' 'Tsallis' 'TsallisNorm') ";
// "[qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
// "[save images] ('-s' for saving, Null, for not.) "<< std::endl;
//

// Software Guide : BeginCodeSnippet
#include "itkImageRegistrationMethodv4.h"
#include "itkMachadoMutualInformationImageToImageMetricv4.h"
#include "itkNormalizedMachadoMutualInformationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkTransformFileWriter.h"

#include "itkTranslationTransform.h"
#include "itkVersorTransform.h"
#include "itkCompositeTransform.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#include "chrono"
#include "math.h"

// Creating txt file
using FileType = std::ofstream;
static FileType execution;
static FileType optimization;

//  at every change of stage and resolution level.

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  using Self = RegistrationInterfaceCommand;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  RegistrationInterfaceCommand() = default;

public:
  using RegistrationType = TRegistration;

  // The Execute function simply calls another version of the \code{Execute()}
  // method accepting a \code{const} input object
  void
  Execute(itk::Object * object, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)object, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    if (!(itk::MultiResolutionIterationEvent().CheckEvent(&event)))
    {
      return;
    }

    std::cout << "\nObserving from class " << object->GetNameOfClass();
    if (!object->GetObjectName().empty())
    {
      std::cout << " \"" << object->GetObjectName() << "\"" << std::endl;
    }

    const auto * registration = static_cast<const RegistrationType *>(object);

    unsigned int currentLevel = registration->GetCurrentLevel();
    typename RegistrationType::ShrinkFactorsPerDimensionContainerType shrinkFactors =
      registration->GetShrinkFactorsPerDimension(currentLevel);
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmas =
      registration->GetSmoothingSigmasPerLevel();

    std::cout << "-------------------------------------" << std::endl;
    // std::cout <<registration->Get'<< std::endl;
    std::cout << " Current multi-resolution level = " << currentLevel << std::endl;
    std::cout << "    shrink factor = " << shrinkFactors << std::endl;
    std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel] << std::endl;
    std::cout << std::endl;
  }
};

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"
class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};

public:
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef   const OptimizerType *                            OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
    OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;

    execution <<optimizer->GetCurrentIteration()<<","<<optimizer->GetValue()<<std::endl;
    }
};

const    unsigned int    Dimension = 3;
typedef  float           PixelType;

typedef itk::Image< PixelType, Dimension >  FixedImageType;
typedef itk::Image< PixelType, Dimension >  MovingImageType;

//using TransformType = itk::VersorRigid3DTransform< double >;
using RotationTransformType = itk::VersorTransform< double >;
using TranslationTransformType = itk::TranslationTransform< double >;
using CompositeTransformType = itk::CompositeTransform<double, Dimension >;


typedef itk::RegularStepGradientDescentOptimizerv4<double>     OptimizerType;
typedef itk::ImageRegistrationMethodv4<
                                  FixedImageType,
                                  MovingImageType,
                                  RotationTransformType    > RotationRegistrationType;
typedef itk::ImageRegistrationMethodv4<
                                  FixedImageType,
                                  MovingImageType,
                                  TranslationTransformType    > TranslationRegistrationType;
typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

int SaveImages ( FixedImageType::Pointer fixedImage,
                 MovingImageType::Pointer movingImage,
                 CompositeTransformType::Pointer finalTransform,
                 std::string qValueStringRotation, std::string qValueStringTranslation){
    //..............................................................
    // Writing OUTPUT images

    typedef itk::ResampleImageFilter<
            MovingImageType,
            FixedImageType >    ResampleFilterType;

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform( finalTransform );
    resample->SetInput( movingImage );

    PixelType defaultPixelValue = 0;

    // Seting aditional resampling information.
    resample->SetSize(  fixedImage->GetLargestPossibleRegion().GetSize() );
    resample->SetOutputOrigin(  fixedImage->GetOrigin() );
    resample->SetOutputSpacing( fixedImage->GetSpacing() );
    resample->SetOutputDirection( fixedImage->GetDirection() );
    resample->SetDefaultPixelValue( defaultPixelValue );

    typedef  float  OutputPixelType;
    const unsigned int Dimension = 3;

    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

    typedef itk::CastImageFilter<
            FixedImageType,
            OutputImageType > CastFilterType;

    typedef itk::ImageFileWriter< OutputImageType >  WriterType;

    WriterType::Pointer      writer =  WriterType::New();
    CastFilterType::Pointer  caster =  CastFilterType::New();

    writer->SetFileName("RegisteredImage_q=" + qValueStringRotation + "-" + qValueStringTranslation + ".nrrd");

    caster->SetInput( resample->GetOutput() );
    writer->SetInput( caster->GetOutput()   );
    writer->Update();

    // Generate checkerboards before and after registration
    //
    typedef itk::CheckerBoardImageFilter< FixedImageType > CheckerBoardFilterType;

    CheckerBoardFilterType::Pointer checker = CheckerBoardFilterType::New();

    // Inputting the fixed image and the resampler filter
    // inputted with the moving image.
    // By changing the transform, one can export before or after
    // registration results.

    checker->SetInput1( fixedImage );
    checker->SetInput2( resample->GetOutput() );

    caster->SetInput( checker->GetOutput() );
    writer->SetInput( caster->GetOutput()  );

    resample->SetDefaultPixelValue( 0 );

    // Before registration ================
    // It will set the identity transform to the moving image
    // resampling at - resample filter - .

    CompositeTransformType::Pointer identityTransformComposed = CompositeTransformType::New();
    RotationTransformType::Pointer identityTransformRotation = RotationTransformType::New();
    TranslationTransformType::Pointer identityTransformTranslation = TranslationTransformType::New();
    identityTransformRotation->SetIdentity();
    identityTransformTranslation->SetIdentity();
    identityTransformComposed->AddTransform(identityTransformTranslation);
    identityTransformComposed->AddTransform(identityTransformRotation);
    resample->SetTransform( identityTransformComposed );

    writer->SetFileName( "CheckBoardBefore_q=" + qValueStringRotation + "-" + qValueStringTranslation + ".nrrd" );
    writer->Update();

    // After registration =================
    // Set the last transformation obtainned in the registrations executions

    resample->SetTransform( finalTransform );
    writer->SetFileName( "CheckBoardAfter_q=" + qValueStringRotation + "-" + qValueStringTranslation + ".nrrd" );
    writer->Update();

    std::cout<<"Images saved!"<<std::endl;

    // .............................................................
    // Writing transform

    // Writing Transform
    using TransformWriterType = itk::TransformFileWriter;
    TransformWriterType::Pointer transformWriter = TransformWriterType::New();
    transformWriter->SetInput(finalTransform);
    transformWriter->SetFileName("finalTransform_q=" + qValueStringRotation + "-" + qValueStringTranslation + ".tfm");
    transformWriter->Update();

    // .............................................................

      using VectorPixelType = itk::Vector<float, Dimension>;
      using DisplacementFieldImageType = itk::Image<VectorPixelType, Dimension>;

      using DisplacementFieldGeneratorType =
        itk::TransformToDisplacementFieldFilter<DisplacementFieldImageType,
    double>;

      // Create an setup displacement field generator.
      DisplacementFieldGeneratorType::Pointer dispfieldGenerator =
        DisplacementFieldGeneratorType::New();
      dispfieldGenerator->UseReferenceImageOn();
      dispfieldGenerator->SetReferenceImage(fixedImage);
      dispfieldGenerator->SetTransform(finalTransform);
      try
      {
        dispfieldGenerator->Update();
      }
      catch (itk::ExceptionObject & err)
      {
        std::cerr << "Exception detected while generating deformation field";
        std::cerr << " : " << err << std::endl;
        return EXIT_FAILURE;
      }

      using FieldWriterType = itk::ImageFileWriter<DisplacementFieldImageType>;
      FieldWriterType::Pointer fieldWriter = FieldWriterType::New();

      fieldWriter->SetInput(dispfieldGenerator->GetOutput());

      fieldWriter->SetFileName("DisplacementField_q=" + qValueStringRotation +"-" + qValueStringTranslation + ".nrrd");
      try
      {
        fieldWriter->Update();
      }
      catch (itk::ExceptionObject & excp)
      {
        std::cerr << "Exception thrown " << std::endl;
        std::cerr << excp << std::endl;
        return EXIT_FAILURE;
      }
      std::cout<<"Deformation Vector Field and Transform Saved!"<<std::endl;

      return EXIT_SUCCESS;
}

int main( int argc, char *argv[] )
{
  if( argc < 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0] << std::endl;
    std::cerr << " fixedImageFile   movingImageFile  MetricType ('Mattes' 'Tsallis' 'TsallisNorm') ";
    std::cerr <<  "[qValue] [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
    std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
    return EXIT_FAILURE;
    }

  CompositeTransformType::Pointer compositeTransform = CompositeTransformType::New();

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  argv[1] );
  movingImageReader->SetFileName( argv[2] );

  // Configuring qValue and output-file
  //
  // type var holds the metric type to be used.
  // strategy says if it will optimize q (true) or single execution (false)
  std::string type = argv[3];
  std::cout<<type<<" metric choosen! "<<std::endl;
  std::cout<<" "<<std::endl;

  double qValueR = 1.0;
  double qValueT = 1.0;
  std::string strategy;

  if (type == "Tsallis" || type == "TsallisNorm" ){
      if( argc > 5 ){

          qValueR = atof(argv[4]);
          qValueT = atof(argv[5]);

          // argv[5] is a bool var to indicate q-optimization or single execution.
          strategy = argv[6];

          if(strategy == "-o"){

              // strategy = -o -> will perform an optimization of q-value:
              std::cout<<"q-optimization routine choosen! "<<std::endl;
              std::cout<<std::endl;
              std::string fileName = type + "_Optimization.csv";
              optimization.open (fileName);
              std::string metricType = type + "MetricValue";
              optimization <<"q-value,MetricValue"<<std::endl;
          }else if(strategy == "-e"){

              // strategy = -o -> will perform a single execution:
              std::cout<<"Execution routine choosen! "<<std::endl;
              std::cout<<std::endl;
              std::stringstream qValueStringRotation;
              qValueStringRotation << std::fixed << std::setprecision(2) << qValueR;
              std::stringstream qValueStringTranslation;
              qValueStringTranslation << std::fixed << std::setprecision(2) << qValueT;
              std::string fileName = type + "_Execution_q=" + qValueStringRotation.str()+ "-"+ qValueStringTranslation.str() + ".csv";
              execution.open (fileName);
              execution <<"iterations,metric_value"<<std::endl;
          }

      } else {

          std::cerr << "Missing Parameters " << std::endl;
          std::cerr << " fixedImageFile   movingImageFile  MetricType ('Mattes' 'Tsallis' 'TsallisNorm') ";
          std::cerr <<  "[qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
          std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
          return EXIT_FAILURE;
      }
  } else if ( type == "Mattes" ) {

      std::cout<<"Execution routine choosen! "<<std::endl;
      std::cout<<" "<<std::endl;
      execution.open ("Mattes_Execution.csv");
      execution <<"iterations,metric_value"<<std::endl;
  } else {

      std::cerr << "Incorrect Parameters " << std::endl;
      std::cerr << " fixedImageFile   movingImageFile  MetricType ('Mattes' 'Tsallis' 'TsallisNorm') ";
      std::cerr <<  "[qR] [qT]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
      std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
      return EXIT_FAILURE;
  }

  for (double qR = 0.1; qR <= qValueR; qR += 0.1){
      for (double qT = 0.1; qT <= qValueT; qT += 0.1){

          if (strategy == "-e" ){
              // meaning is a single execution with a q-metric
              qR = qValueR;
              qT = qValueT;           
          }
          if (type == "Mattes" ){
              // meaning is a single execution with a q-metric
              qR = 1.00;
              qT = 1.00;
          }  

          RotationRegistrationType::Pointer   registrationR  = RotationRegistrationType::New();
          TranslationRegistrationType::Pointer   registrationT  = TranslationRegistrationType::New();
          OptimizerType::Pointer       optimizerR    = OptimizerType::New();
          OptimizerType::Pointer       optimizerT    = OptimizerType::New();
          registrationR->SetOptimizer(     optimizerR     );
          registrationT->SetOptimizer(     optimizerT     );

          // Metric check configuration;
          //
          unsigned int numberOfBins = 50;

          // Choosing the metric type.
          if (type == "Tsallis"){
              
              if ( qR == 1.00 || qT == 1.00 ){
                // Will use Mattes metric 
                goto mattes;
              } 

              typedef itk::MachadoMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > TsallisMetricType;
              TsallisMetricType::Pointer tsallisMetricR = TsallisMetricType::New();
              TsallisMetricType::Pointer tsallisMetricT = TsallisMetricType::New();

              tsallisMetricR->SetqValue(qR);
              tsallisMetricR->SetNumberOfHistogramBins( numberOfBins );
              tsallisMetricR->SetUseMovingImageGradientFilter( false );
              tsallisMetricR->SetUseFixedImageGradientFilter( false );
              tsallisMetricR->SetUseSampledPointSet(false);
              registrationR->SetMetric( tsallisMetricR );

              tsallisMetricT->SetqValue(qT);
              tsallisMetricT->SetNumberOfHistogramBins( numberOfBins );
              tsallisMetricT->SetUseMovingImageGradientFilter( false );
              tsallisMetricT->SetUseFixedImageGradientFilter( false );
              tsallisMetricT->SetUseSampledPointSet(false);
              registrationT->SetMetric( tsallisMetricT );
          }
          else if (type == "TsallisNorm"){

              if (qR == 1.00 || qT == 1.00){
                // Will use Mattes metric 
                goto mattes;
              } 

              typedef itk::NormalizedMachadoMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > TsallisNormMetricType;
              TsallisNormMetricType::Pointer tsallisNormMetricR = TsallisNormMetricType::New();
              TsallisNormMetricType::Pointer tsallisNormMetricT = TsallisNormMetricType::New();
              
              tsallisNormMetricR->SetqValue(qR);
              tsallisNormMetricR->SetNumberOfHistogramBins( numberOfBins );
              tsallisNormMetricR->SetUseMovingImageGradientFilter( false );
              tsallisNormMetricR->SetUseFixedImageGradientFilter( false );
              tsallisNormMetricR->SetUseSampledPointSet(false);
              registrationR->SetMetric( tsallisNormMetricR  );

              tsallisNormMetricT->SetqValue(qT);
              tsallisNormMetricT->SetNumberOfHistogramBins( numberOfBins );
              tsallisNormMetricT->SetUseMovingImageGradientFilter( false );
              tsallisNormMetricT->SetUseFixedImageGradientFilter( false );
              tsallisNormMetricT->SetUseSampledPointSet(false);
              registrationT->SetMetric( tsallisNormMetricT  );
          }
          else if (type == "Mattes"){  
              //qR, qT = 1.00, 1.00;              
              mattes:
              
              typedef itk::MattesMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > MattesMetricType;
              MattesMetricType::Pointer  mattesMetric = MattesMetricType::New();
              mattesMetric->SetNumberOfHistogramBins( numberOfBins );
              mattesMetric->SetUseMovingImageGradientFilter( false );
              mattesMetric->SetUseFixedImageGradientFilter( false );
              mattesMetric->SetUseSampledPointSet(false);

              if (type == "Mattes"){
                
                std::cout<<"Mattes Metric Set."<<std::endl;
                registrationR->SetMetric( mattesMetric  );
                registrationT->SetMetric( mattesMetric  );
              
              }
              if (type == "Tsallis" || type == "TsallisNorm"){
                if (qR == 1.00){
                  
                  std::cout<<"Tsallis or Tsallis norm using Mattes for qR = 1.00.";
                  registrationR->SetMetric( mattesMetric  );
                
                }
                if (qT == 1.00 ){

                  std::cout<<"Tsallis or Tsallis norm using Mattes for qT = 1.00.";
                  registrationT->SetMetric( mattesMetric  );

                }
              }
              
          }
          
          std::cout<< "qR = "<<qR<< "  qT = "<<qT<<std::endl;

          //*****************************************************
          // Translation Stage;
          std::cout<<"Translation Stage***********************************"<<std::endl;
          
          registrationT->SetFixedImage(    fixedImageReader->GetOutput()    );
          registrationT->SetMovingImage(   movingImageReader->GetOutput()   );
    
          TranslationTransformType::Pointer  initialTransformTranslation = TranslationTransformType::New();
          initialTransformTranslation->SetIdentity();

          //using TranslationTransformInitializerType = itk::CenteredTransformInitializer<
          //    TranslationTransformType,
          //    FixedImageType,
          //    MovingImageType >;

          //TranslationTransformInitializerType::Pointer translationInitializer = TranslationTransformInitializerType::New();

          /*
          translationInitializer->SetTransform(  initialTransformTranslation );
          translationInitializer->SetFixedImage(  fixedImageReader->GetOutput() );
          translationInitializer->SetMovingImage(  movingImageReader->GetOutput() );
          translationInitializer->MomentsOn();
          translationInitializer->InitializeTransform();
          */

          registrationT->SetInitialTransform( initialTransformTranslation );
          registrationT->InPlaceOn();

          // Parameter scale setter
          //
          using OptimizerScalesType = OptimizerType::ScalesType;
          OptimizerScalesType optimizerScalesTranslation ( initialTransformTranslation->GetNumberOfParameters() );
          optimizerScalesTranslation[0] = 1.0;
          optimizerScalesTranslation[1] = 1.0;
          optimizerScalesTranslation[2] = 1.0;
          optimizerT->SetScales( optimizerScalesTranslation );

          // Cinfiguring the optimizer
          // Different line

          optimizerT->SetLearningRate( 1.0 );
          optimizerT->SetMinimumStepLength( 0.01 );
          optimizerT->SetNumberOfIterations( 300 );
          optimizerT->ReturnBestParametersAndValueOn();

          // Create the Command observer and register it with the optimizer.
          //
          CommandIterationUpdate::Pointer observerTranslation = CommandIterationUpdate::New();
          optimizerT->AddObserver( itk::IterationEvent(), observerTranslation );

          const unsigned int numberOfLevels = 1;

          TranslationRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevelTranslation;
          shrinkFactorsPerLevelTranslation.SetSize( numberOfLevels );
          shrinkFactorsPerLevelTranslation[0] = 1;
          //shrinkFactorsPerLevelTranslation[1] = 2;
          //shrinkFactorsPerLevelTranslation[2] = 1;
          
          TranslationRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevelTranslation;
          smoothingSigmasPerLevelTranslation.SetSize( numberOfLevels );
          smoothingSigmasPerLevelTranslation[0] = 0;
          //smoothingSigmasPerLevelTranslation[1] = 1;
          //smoothingSigmasPerLevelTranslation[2] = 0;
          
          registrationT->SetNumberOfLevels ( numberOfLevels );
          registrationT->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevelTranslation );
          registrationT->SetShrinkFactorsPerLevel( shrinkFactorsPerLevelTranslation );

          using TranslationCommandRegistrationType = RegistrationInterfaceCommand<TranslationRegistrationType>;
          TranslationCommandRegistrationType::Pointer commandTranslation = TranslationCommandRegistrationType::New();
          registrationT->AddObserver(itk::MultiResolutionIterationEvent(), commandTranslation);

          try
            {
            registrationT->Update();
            std::cout << std::endl;
            std::cout << "Optimizer stop condition: "
                      << registrationT->GetOptimizer()->GetStopConditionDescription()
                      << std::endl;
            }
          catch( itk::ExceptionObject & err )
            {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
            }

            compositeTransform->AddTransform(initialTransformTranslation);

          //*****************************************************
          // Rotation Stage;
          std::cout<<"Rotation Stage***********************************"<<std::endl;
          
          registrationR->SetFixedImage(    fixedImageReader->GetOutput()    );
          registrationR->SetMovingImage(   movingImageReader->GetOutput()   );

          // Setting initial transform configuration::

          using RotationTransformInitializerType = itk::CenteredTransformInitializer<
              RotationTransformType,
              FixedImageType,
              MovingImageType >;
          
          RotationTransformType::Pointer initialTransformRotation = RotationTransformType::New();
          initialTransformRotation->SetIdentity();

          RotationTransformInitializerType::Pointer rotationInitializer = RotationTransformInitializerType::New();
          rotationInitializer->SetTransform(  initialTransformRotation );
          rotationInitializer->SetFixedImage(  fixedImageReader->GetOutput() );
          rotationInitializer->SetMovingImage(  movingImageReader->GetOutput() );
          rotationInitializer->MomentsOn();
          rotationInitializer->InitializeTransform();
          

          // Angular componet of initial transform
          //
          using VersorType = RotationTransformType::VersorType;
          using VectorType = VersorType::VectorType;
          VersorType     rotation;
          VectorType     axis;
          axis[0] = 0.0;
          axis[1] = 0.0;
          axis[2] = 1.0;
          constexpr double angle = 0;
          rotation.Set(  axis, angle  );
          initialTransformRotation->SetRotation( rotation );
          
          registrationR->SetInitialTransform( initialTransformRotation );
          registrationR->InPlaceOn();

          // Connecting previous stage transform to the next stage;
          registrationR->SetMovingInitialTransform(
            initialTransformTranslation);

          OptimizerScalesType optimizerScalesRotation ( initialTransformRotation->GetNumberOfParameters() );      
          const double rotationScale = 1.0 / 1000.0;
          optimizerScalesRotation[0] = rotationScale;
          optimizerScalesRotation[1] = rotationScale;
          optimizerScalesRotation[2] = rotationScale;
          optimizerR->SetScales( optimizerScalesRotation );

          optimizerR->SetLearningRate( 1.0 );
          optimizerR->SetMinimumStepLength( 0.01 );
          optimizerR->SetNumberOfIterations( 300 );
          optimizerR->ReturnBestParametersAndValueOn();

          CommandIterationUpdate::Pointer observerRotation = CommandIterationUpdate::New();
          optimizerR->AddObserver( itk::IterationEvent(), observerRotation );

          // One level registration process without shrinking and smoothing.
          //
          const unsigned int numberOfLevelsR = 3;

          RotationRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevelRotation;
          shrinkFactorsPerLevelRotation.SetSize( numberOfLevelsR );
          shrinkFactorsPerLevelRotation[0] = 3;
          shrinkFactorsPerLevelRotation[1] = 2;
          shrinkFactorsPerLevelRotation[2] = 1;

          RotationRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevelRotation;
          smoothingSigmasPerLevelRotation.SetSize( numberOfLevelsR );
          smoothingSigmasPerLevelRotation[0] = 2;
          smoothingSigmasPerLevelRotation[1] = 1;
          smoothingSigmasPerLevelRotation[2] = 0;

          registrationR->SetNumberOfLevels ( numberOfLevelsR );
          registrationR->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevelRotation );
          registrationR->SetShrinkFactorsPerLevel( shrinkFactorsPerLevelRotation );

          using RotationCommandRegistrationType = RegistrationInterfaceCommand<RotationRegistrationType>;
          RotationCommandRegistrationType::Pointer commandRotation = RotationCommandRegistrationType::New();
          registrationR->AddObserver(itk::MultiResolutionIterationEvent(), commandRotation);
          
          try
            {
            registrationR->Update();
            std::cout << std::endl;
            std::cout << "Optimizer stop condition: "
                      << registrationR->GetOptimizer()->GetStopConditionDescription()
                      << std::endl;
            }
          catch( itk::ExceptionObject & err )
            {
            std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
            }  
          // OUTPUTTING RESULTS

          compositeTransform->AddTransform(initialTransformRotation);  

          //
          //TransformType::ParametersType finalParameters =
          //                          registration->GetOutput()->Get()->GetParameters();

          // For stability reasons it may be desirable to round up the values of translation
          //
          // unsigned long numberOfIterations = optimizer->GetCurrentIteration();

          RotationTransformType::Pointer finalTransform = initialTransformRotation;
          double metricValue = optimizerR->GetValue();

          /*
          finalTransform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
          finalTransform->SetParameters( finalParameters );
          TransformType::MatrixType matrix = finalTransform->GetMatrix();
          TransformType::OffsetType offset = finalTransform->GetOffset();
          std::cout << "Matrix = " << std::endl << matrix << std::endl;
          std::cout << "Offset = " << std::endl << offset << std::endl; */

          if ( strategy == "-e" && type != "Mattes" ){
              
              // Printing out results
              // Initializing the save flag.
              //
              std::string save;
              save = argv[7];

              if (save == "-s") {
                std::stringstream qValueStringTranslation;
                qValueStringTranslation << std::fixed << std::setprecision(2) << qT; 
                std::stringstream qValueStringRotation;
                qValueStringRotation << std::fixed << std::setprecision(2) << qR;
                SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), compositeTransform, qValueStringRotation.str(),  qValueStringTranslation.str());

              } else {
                std::cout<<"Images not saved!" <<std::endl;
                std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
              }

              break;

          } else if ( type == "Mattes" ) {
              
              // Printing out results
              // Initializing the save flag.
              //
              std::string save;
              save = argv[4];

              if (save == "-s") {
              
                std::stringstream qValueStringTranslation;
                qValueStringTranslation << std::fixed << std::setprecision(2) << qT; 
                std::stringstream qValueStringRotation;
                qValueStringRotation << std::fixed << std::setprecision(2) << qR;
                SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), compositeTransform, qValueStringRotation.str(),  qValueStringTranslation.str());
              
              } else {
                std::cout<<"Images not saved!" <<std::endl;
                std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
              }

              break;          
          }
          
          optimization <<qR<<","<<qT<<","<<metricValue<<std::endl;
          
          // Printing out results
          // Initializing the save flag.
          //
          std::string save;
          save = argv[7];

          if (save == "-s") {
              std::stringstream qValueStringTranslation;
              qValueStringTranslation << std::fixed << std::setprecision(2) << qT; 
              std::stringstream qValueStringRotation;
              qValueStringRotation << std::fixed << std::setprecision(2) << qR;
              SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), compositeTransform, qValueStringRotation.str(),  qValueStringTranslation.str());
              
          } else {
            std::cout<<"Images not saved!" <<std::endl;
            std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
          }
      } // q loop;
    }// q loop;
  return EXIT_SUCCESS;
}
