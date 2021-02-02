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
#include "itkTranslationTransform.h"
#include "itkMachadoMutualInformationImageToImageMetricv4.h"
#include "itkNormalizedMachadoMutualInformationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkTransformFileWriter.h"

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

using TransformType = itk::VersorRigid3DTransform< double >;
typedef itk::RegularStepGradientDescentOptimizerv4<double>     OptimizerType;
typedef itk::ImageRegistrationMethodv4<
                                  FixedImageType,
                                  MovingImageType,
                                  TransformType    > RegistrationType;
typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

int SaveImages ( FixedImageType::Pointer fixedImage,
                 MovingImageType::Pointer movingImage,
                 TransformType::Pointer finalTransform,
                 std::string qValueStr){
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

    writer->SetFileName("RegisteredImage_q=" + qValueStr + ".nrrd");

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

    TransformType::Pointer identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    resample->SetTransform( identityTransform );

    writer->SetFileName( "CheckBoardBefore_q=" + qValueStr + ".nrrd" );
    writer->Update();

    // After registration =================
    // Set the last transformation obtainned in the registrations executions

    resample->SetTransform( finalTransform );
    writer->SetFileName( "CheckBoardAfter_q=" + qValueStr + ".nrrd" );
    writer->Update();

    std::cout<<"Images saved!"<<std::endl;

    // .............................................................
    // Writing transform

    // Writing Transform
    using TransformWriterType = itk::TransformFileWriter;
    TransformWriterType::Pointer transformWriter = TransformWriterType::New();
    transformWriter->SetInput(finalTransform);
    transformWriter->SetFileName("finalTransform_q=" + qValueStr + ".tfm");
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

      fieldWriter->SetFileName("DisplacementField_q=" + qValueStr + ".nrrd");
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
    std::cerr << " [qValue] [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
    std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
    return EXIT_FAILURE;
    }


  // Final Transform object;
  TransformType::Pointer finalTransform = TransformType::New();


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

  double qValue = 1.0;
  std::string strategy;

  if (type == "Tsallis" || type == "TsallisNorm" ){
      if( argc > 5 ){

          qValue = atof(argv[4]);

          // argv[5] is a bool var to indicate q-optimization or single execution.
          strategy = argv[5];

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
              std::stringstream qValueString;
              qValueString << std::fixed << std::setprecision(2) << qValue;
              std::string fileName = type + "_Execution_q=" + qValueString.str() + ".csv";
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
      std::cerr << " [qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
      std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
      return EXIT_FAILURE;
  }


  for (double q = 0.01; q < qValue; q += 0.01){

      if (strategy == "-e" ){
          // meaning is a single execution with a q-metric. Before repeating the loop the code break.
          q = qValue;
          // std::cout<< "q-Value = "<<q<<std::endl;
      }

      std::cout<< "q-Value = "<<q<<std::endl;

      RegistrationType::Pointer   registration  = RegistrationType::New();

      OptimizerType::Pointer       optimizer    = OptimizerType::New();
      registration->SetOptimizer(     optimizer     );

      // Metric check configuration;
      //
      unsigned int numberOfBins = 50;

      std::cout<<"Number of Bins = "<<numberOfBins<<std::endl;

      // Choosing the metric type.
      if (type == "Tsallis"){
          // for q = 1.0 one should use Mattes Entropy
          if (q == 1.0 ) {
              goto mattes;
          }
          typedef itk::MachadoMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > TsallisMetricType;
          TsallisMetricType::Pointer tsallisMetric = TsallisMetricType::New();

          tsallisMetric->SetqValue(q);

          tsallisMetric->SetNumberOfHistogramBins( numberOfBins );
          tsallisMetric->SetUseMovingImageGradientFilter( false );
          tsallisMetric->SetUseFixedImageGradientFilter( false );
          tsallisMetric->SetUseSampledPointSet(false);

          registration->SetMetric( tsallisMetric );
      }
      else if (type == "TsallisNorm"){
          // for q = 1.0 one should use Mattes Entropy
          if (q == 1.0 ) {
              goto mattes;
          }

          typedef itk::NormalizedMachadoMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > TsallisNormMetricType;
          TsallisNormMetricType::Pointer tsallisNormMetric = TsallisNormMetricType::New();

          tsallisNormMetric->SetqValue(q);

          tsallisNormMetric->SetNumberOfHistogramBins( numberOfBins );
          tsallisNormMetric->SetUseMovingImageGradientFilter( false );
          tsallisNormMetric->SetUseFixedImageGradientFilter( false );
          tsallisNormMetric->SetUseSampledPointSet(false);

          registration->SetMetric( tsallisNormMetric  );
      }
      else if (type == "Mattes"){
          mattes:
          std::cout<<"Using Mattes Entropy Class."<<std::endl;

          typedef itk::MattesMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > MattesMetricType;
          MattesMetricType::Pointer  mattesMetric = MattesMetricType::New();

          mattesMetric->SetNumberOfHistogramBins( numberOfBins );
          mattesMetric->SetUseMovingImageGradientFilter( false );
          mattesMetric->SetUseFixedImageGradientFilter( false );
          mattesMetric->SetUseSampledPointSet(false);

          registration->SetMetric( mattesMetric  );
      }

      registration->SetFixedImage(    fixedImageReader->GetOutput()    );
      registration->SetMovingImage(   movingImageReader->GetOutput()   );

      // Setting initial transform configuration::
      //
      TransformType::Pointer  initialTransform = TransformType::New();


      using TransformInitializerType = itk::CenteredTransformInitializer<
          TransformType,
          FixedImageType,
          MovingImageType >;
          TransformInitializerType::Pointer initializer =
          TransformInitializerType::New();

      initializer->SetTransform(  initialTransform );
      initializer->SetFixedImage(  fixedImageReader->GetOutput() );
      initializer->SetMovingImage(  movingImageReader->GetOutput() );
      initializer->MomentsOn();

      initializer->InitializeTransform();

      // Angular componet of initial transform
      //
      using VersorType = TransformType::VersorType;
      using VectorType = VersorType::VectorType;
      VersorType     rotation;
      VectorType     axis;
      axis[0] = 0.0;
      axis[1] = 0.0;
      axis[2] = 1.0;
      constexpr double angle = 0;
      rotation.Set(  axis, angle  );
      initialTransform->SetRotation( rotation );

      registration->SetInitialTransform( initialTransform );

      /*
      TransformType::Pointer  initialTransform0 = initialTransform;
      initialTransform0->SetIdentity();
      std::cout<<"Saving raw DVF to assess initial difference."<<std::endl;
      std::stringstream qValueString;
      qValueString << std::fixed << std::setprecision(2) << q;
      SaveImages(fixedImageReader->GetOutput(),movingImageReader->GetOutput(),initialTransform0, qValueString.str());
      std::cout<<"Saved."<<std::endl;
      //break; */

      // Parameter scale setter
      //
      using OptimizerScalesType = OptimizerType::ScalesType;
      OptimizerScalesType optimizerScales( initialTransform->GetNumberOfParameters() );
      const double translationScale = 1.0 / 1000.0;
      optimizerScales[0] = 1.0;
      optimizerScales[1] = 1.0;
      optimizerScales[2] = 1.0;
      optimizerScales[3] = translationScale;
      optimizerScales[4] = translationScale;
      optimizerScales[5] = translationScale;
      optimizer->SetScales( optimizerScales );

      // Cinfiguring the optimizer
      // Different line

      optimizer->SetLearningRate( 1.0 );
      optimizer->SetMinimumStepLength( 0.001 );
      optimizer->SetNumberOfIterations( 300 );
      optimizer->ReturnBestParametersAndValueOn();
      // optimizer->SetGradientMagnitudeTolerance(0.0001);

      // Create the Command observer and register it with the optimizer.
      //
      CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
      optimizer->AddObserver( itk::IterationEvent(), observer );

      // One level registration process without shrinking and smoothing.
      //
      const unsigned int numberOfLevels = 3;

      RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
      shrinkFactorsPerLevel.SetSize( numberOfLevels );
      shrinkFactorsPerLevel[0] = 3;
      shrinkFactorsPerLevel[1] = 2;
      shrinkFactorsPerLevel[2] = 1;

      RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
      smoothingSigmasPerLevel.SetSize( numberOfLevels );
      smoothingSigmasPerLevel[0] = 2;
      smoothingSigmasPerLevel[1] = 1;
      smoothingSigmasPerLevel[2] = 0;

      registration->SetNumberOfLevels ( numberOfLevels );
      registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
      registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

      using RigidCommandRegistrationType = RegistrationInterfaceCommand<RegistrationType>;
      RigidCommandRegistrationType::Pointer command = RigidCommandRegistrationType::New();
      registration->AddObserver(itk::MultiResolutionIterationEvent(), command);

      try
        {
        registration->Update();
        std::cout << std::endl;
        std::cout << "Optimizer stop condition: "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
        }
      catch( itk::ExceptionObject & err )
        {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
        }

      // OUTPUTTING RESULTS
      //
      TransformType::ParametersType finalParameters =
                                registration->GetOutput()->Get()->GetParameters();

      // For stability reasons it may be desirable to round up the values of translation
      //
      // unsigned long numberOfIterations = optimizer->GetCurrentIteration();

      finalTransform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
      finalTransform->SetParameters( finalParameters );

      double metricValue = optimizer->GetValue();

      TransformType::MatrixType matrix = finalTransform->GetMatrix();
      TransformType::OffsetType offset = finalTransform->GetOffset();
      std::cout << "Matrix = " << std::endl << matrix << std::endl;
      std::cout << "Offset = " << std::endl << offset << std::endl;

      if ( strategy == "-e" && type != "Mattes" ){
          
          // Printing out results
          // Initializing the save flag.
          //
          std::string save;
          save = argv[6];

          if (save == "-s") {
             std::stringstream qValueString;
             qValueString << std::fixed << std::setprecision(2) << q;
             SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), finalTransform, qValueString.str());

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
             std::stringstream qValueString;
             qValueString << std::fixed << std::setprecision(2) << q;
             SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), finalTransform, qValueString.str());

          } else {
             std::cout<<"Images not saved!" <<std::endl;
             std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
          }

          break;          
      }
      
      optimization <<q<<","<<metricValue<<std::endl;
      
      // Printing out results
      // Initializing the save flag.
      //
      std::string save;
      save = argv[6];

      if (save == "-s") {
          std::stringstream qValueString;
          qValueString << std::fixed << std::setprecision(2) << q;
          SaveImages(fixedImageReader->GetOutput(), movingImageReader->GetOutput(), finalTransform, qValueString.str());

      } else {
         std::cout<<"Images not saved!" <<std::endl;
         std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
      }
   } // q-value loop;
  
  return EXIT_SUCCESS;
}
