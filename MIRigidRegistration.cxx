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

#include <iostream>
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


int main( int argc, char *argv[] )
{
  if( argc < 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0] << std::endl;
    std::cerr << " fixedImageFile   movingImageFile  MetricType ('Mattes' 'Tsallis' 'TsallisNorm') ";
    std::cerr <<  "[qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
    std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
    return EXIT_FAILURE;
    }

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

  // registration pointer for passage;
  RegistrationType::Pointer registrationPass;

  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

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
              optimization <<"q-value,iterations,displacementVector,displacementError,angle,angleError"<<std::endl;        
          }else if(strategy == "-e"){

              // strategy = -o -> will perform a single execution:
              std::cout<<"Execution routine choosen! "<<std::endl;
              std::cout<<std::endl;
              std::string fileName = type + "_Execution_q=" + std::to_string(qValue) + ".csv";
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
      std::cerr <<  "[qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl;
      std::cerr << " [save images] ('-s' for saving, Null, for not.) "<< std::endl;
      return EXIT_FAILURE;
  }


  for (double q = 0; q <= qValue; q += 0.01){

      if (strategy == "-e" ){
          // meaning is a single execution with a q-metric
          q = qValue;
      }

      RegistrationType::Pointer   registration  = RegistrationType::New();
      registrationPass = registration;

      OptimizerType::Pointer       optimizer    = OptimizerType::New();
      registration->SetOptimizer(     optimizer     );

      // Metric check configuration;
      //
      unsigned int numberOfBins = 50;

      // Choosing the metric type.
      if (type == "Tsallis"){
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

      initializer->SetTransform(   initialTransform );
      initializer->SetFixedImage(  fixedImageReader->GetOutput() );
      initializer->SetMovingImage( movingImageReader->GetOutput() );
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

      optimizer->SetLearningRate( 0.1 );
      optimizer->SetMinimumStepLength( 0.001 );
      optimizer->SetNumberOfIterations( 300 );
      optimizer->ReturnBestParametersAndValueOn();

      // Create the Command observer and register it with the optimizer.
      //
      CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
      optimizer->AddObserver( itk::IterationEvent(), observer );

      // One level registration process without shrinking and smoothing.
      //
      const unsigned int numberOfLevels = 2;

      RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
      shrinkFactorsPerLevel.SetSize( numberOfLevels );
      shrinkFactorsPerLevel[0] = 3;
      shrinkFactorsPerLevel[1] = 2;

      RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
      smoothingSigmasPerLevel.SetSize( numberOfLevels );
      smoothingSigmasPerLevel[0] = 1;
      smoothingSigmasPerLevel[1] = 0;

      registration->SetNumberOfLevels ( numberOfLevels );
      registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
      registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

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
      unsigned long numberOfIterations = optimizer->GetCurrentIteration();

      TransformType::Pointer finalTransform = TransformType::New();
      finalTransform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
      finalTransform->SetParameters( finalParameters );

      TransformType::MatrixType matrix = finalTransform->GetMatrix();
      TransformType::OffsetType offset = finalTransform->GetOffset();
      std::cout << "Matrix = " << std::endl << matrix << std::endl;
      std::cout << "Offset = " << std::endl << offset << std::endl;

      // Calculate the relative error considering the expected values for x, y, and z components
      //
      double angleDegree = std::asin(matrix[0][1]) * (180.0/3.141592653589793238463);
      //std::cout << "Angle = " << angleDegree << std::endl;

      double displacementVector = sqrt(offset[0]*offset[0] + offset[2]*offset[2] + offset[3]*offset[3]);

      double displacementError = 15.0 - displacementVector;
      double angleError = 10.0 - angleDegree;


      if (type == "Mattes" || strategy == "-e"){

          execution <<"NumberOfIterations = " << numberOfIterations << std::endl;
          execution <<"FinalDisplacement = "  << displacementVector << std::endl;
          execution <<"DisplacementError = "  << displacementError << std::endl;
          execution <<"FinalAngle = "         << angle <<std::endl;
          execution <<"AngleError = "         << angleError << std::endl;
          execution.close();

          std::cout << std::endl;
          std::cout << " Result            "    << std::endl;
          std::cout << " qValue          = "    << q << std::endl;
          std::cout << " Number of Iterations = " << numberOfIterations << std::endl;
          std::cout << " Displacement    = "    << displacementVector << std::endl;
          std::cout << " Displacement Error = " << displacementError << std::endl;
          std::cout << " Angle           = "    << angleDegree << std::endl;
          std::cout << " AngleError      = "    << angleError << std::endl;
          std::cout << std::endl;

          break;

      } else {
          optimization <<q<<","<<numberOfIterations<<","
                      <<displacementVector<<","<<displacementError<<","<<angleDegree<<","<<angleError<<std::endl;

          std::cout << std::endl;
          std::cout << " Result            "    << std::endl;
          std::cout << " qValue          = "    << q << std::endl;
          std::cout << " Number of Iterations = " << numberOfIterations<< std::endl;
          std::cout << " Angle           = "    << angleDegree << std::endl;
          std::cout << " AngleError      = "    << angleError << std::endl;
          std::cout << " Displacement    = "    << displacementVector << std::endl;
          std::cout << " Displacement Error = " << displacementError << std::endl;
          std::cout << std::endl;
      }      
   }


  // Printing out results

  // Initializing the save flag.
  std::string save;
  if (argc > 6){
      save = argv[6];
  }
  if (type == "Mattes" && argc > 4){
      save = argv[4];
  }

  if (save == "-s") {


      // Writing OUTPUT images
      typedef itk::ResampleImageFilter<
              MovingImageType,
              FixedImageType >    ResampleFilterType;

      ResampleFilterType::Pointer resample = ResampleFilterType::New();

      resample->SetTransform( registrationPass->GetTransform() );
      resample->SetInput( movingImageReader->GetOutput() );

      FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

      PixelType defaultPixelValue = 100;

      // Seting aditional resampling information.
      resample->SetSize(  fixedImage->GetLargestPossibleRegion().GetSize() );
      resample->SetOutputOrigin(  fixedImage->GetOrigin() );
      resample->SetOutputSpacing( fixedImage->GetSpacing() );
      resample->SetOutputDirection( fixedImage->GetDirection() );
      resample->SetDefaultPixelValue( defaultPixelValue );


      typedef  float  OutputPixelType;

      typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

      typedef itk::CastImageFilter<
              FixedImageType,
              OutputImageType > CastFilterType;

      typedef itk::ImageFileWriter< OutputImageType >  WriterType;

      WriterType::Pointer      writer =  WriterType::New();
      CastFilterType::Pointer  caster =  CastFilterType::New();

      writer->SetFileName( "OutPut.nrrd" );

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

      writer->SetFileName( "CheckBoardBefore.nrrd" );
      writer->Update();

      // After registration =================
      // Set the last transformation obtainned in the registrations executions

      resample->SetTransform( registrationPass->GetTransform() );
      writer->SetFileName( "CheckBoardAfter.nrrd" );
      writer->Update();

      std::cout<<"Images saved!"<<std::endl;
  } else {

      std::cout<<"Images not saved!" <<std::endl;
      std::cout<<"Pass '-s' for saving images or leave null to not saving."<<std::endl;
  }

  return EXIT_SUCCESS;
}
