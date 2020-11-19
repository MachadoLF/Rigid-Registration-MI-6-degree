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
// with Both Mattes and Tsallis Metric
//

// Software Guide : BeginCodeSnippet
#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
#include "itkMachadoMutualInformationImageToImageMetricv4.h"
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

using myFileType = std::ofstream;
static myFileType myfile;

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

  void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
    {
    OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;

    myfile <<optimizer->GetCurrentIteration()+1<<","<<optimizer->GetValue()<<std::endl;
    }
};


int main( int argc, char *argv[] )
{
  if( argc < 2 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << "[qValue] [nTimes to average]";
    std::cerr << "[useExplicitPDFderivatives ] " << std::endl;
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

  // Defining specific metric
  // Tsallis
  // typedef itk::MachadoMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > MetricType;

  // Mattes
  typedef itk::MattesMutualInformationImageToImageMetricv4< FixedImageType,MovingImageType > MetricType;
  // Software Guide : EndCodeSnippet


  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  argv[1] );
  movingImageReader->SetFileName( argv[2] );


  // itinitiallizing qValue and average times

  double qValue = 0.5;
  if( argc > 2 )
  {
      qValue = atof(argv[3]);
  }

  int nTimes = 1;
  if( argc > 3 )
  {
      nTimes = atoi(argv[4]);
  }

  double meanMetricValue = 0.0;
  double meanNumberOfIterations = 0.0;
  double meanXError = 0.0;
  double meanYError = 0.0;
  double meanZError = 0.0;
  double meanAngleError = 0.0;
  double meanBuildupError = 0.0;

  for (int y = 0; y < nTimes; ++y){

      myfile.open ("performance.csv");
      myfile <<"# The performance was carried with qValue = "<<qValue<<" . Each point is equal to a "<<nTimes<<" performance execution."<< std::endl;
      myfile <<"Iteration,MetricValue"<<std::endl;

      RegistrationType::Pointer   registration  = RegistrationType::New();
      registrationPass = registration;

      OptimizerType::Pointer       optimizer    = OptimizerType::New();

      registration->SetOptimizer(     optimizer     );
      // Setting images

      // Setting Metric
      MetricType::Pointer metric = MetricType::New();
      // metric->SetqValue(qValue);
      registration->SetMetric( metric  );

      //  The metric requires the user to specify the number of bins
      //  used to compute the entropy. In a typical application, 50 histogram bins
      //  are sufficient.

      unsigned int numberOfBins = 24;

      metric->SetNumberOfHistogramBins( numberOfBins );

      metric->SetUseMovingImageGradientFilter( false );
      metric->SetUseFixedImageGradientFilter( false );

      // It will use the whole image to test.
      metric->SetUseSampledPointSet(false);

      registration->SetFixedImage(    fixedImageReader->GetOutput()    );
      registration->SetMovingImage(   movingImageReader->GetOutput()   );

      // Setting initial configuration::

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

      optimizer->SetLearningRate( 0.2 );
      optimizer->SetMinimumStepLength( 0.001 );
      optimizer->SetNumberOfIterations( 300 );
      optimizer->ReturnBestParametersAndValueOn();

      // Create the Command observer and register it with the optimizer.
      //
      CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
      optimizer->AddObserver( itk::IterationEvent(), observer );

      // One level registration process without shrinking and smoothing.
      //
      const unsigned int numberOfLevels = 1;

      RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
      shrinkFactorsPerLevel.SetSize( 1 );
      shrinkFactorsPerLevel[0] = 1;

      RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
      smoothingSigmasPerLevel.SetSize( 1 );
      smoothingSigmasPerLevel[0] = 0;

      registration->SetNumberOfLevels ( numberOfLevels );
      registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
      registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

      // Instead of using the whole virtual domain (usually fixed image domain) for the registration,
      // we can use a spatial sampled point set by supplying an arbitrary point list over which to
      // evaluate the metric.

      /*
      RegistrationType::MetricSamplingStrategyType  samplingStrategy  =
        RegistrationType::RANDOM;

      double samplingPercentage = 0.20;

      // In ITKv4, a single virtual domain or spatial sample point set is used for the
      // all iterations of the registration process. The use of a single sample set results
      // in a smooth cost function that can improve the functionality of
      // the optimizer.
      //
      // The spatial point set is pseudo randomly generated. For
      // reproducible results an integer seed should set.


      registration->SetMetricSamplingStrategy( samplingStrategy );
      registration->SetMetricSamplingPercentage( samplingPercentage );
      registration->MetricSamplingReinitializeSeed( 121213 );
      */

      try
        {
        registration->Update();
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

      TransformType::ParametersType finalParameters =
                                registration->GetOutput()->Get()->GetParameters();

      //double TranslationAlongX = finalParameters[0];
      //double TranslationAlongY = finalParameters[1];

      // For stability reasons it may be desirable to round up the values of translation
      //
      unsigned int numberOfIterations = optimizer->GetCurrentIteration();
                                        // -1.0 is used for correcting the number of executions.
      double metricValue = optimizer->GetCurrentMetricValue();

      meanNumberOfIterations += numberOfIterations;
      meanMetricValue += metricValue;

      TransformType::Pointer finalTransform = TransformType::New();
      finalTransform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
      finalTransform->SetParameters( finalParameters );

      TransformType::MatrixType matrix = finalTransform->GetMatrix();
      TransformType::OffsetType offset = finalTransform->GetOffset();
      std::cout << "Matrix = " << std::endl << matrix << std::endl;
      std::cout << "Offset = " << std::endl << offset << std::endl;
        //std::cout << "Matrix[0][1] = " << matrix[0][1] << std::endl;


        // Calculate the relative error considering the expected values for x and y components
      double angleDegree = std::asin(matrix[0][1]) * (180.0/3.141592653589793238463);
      std::cout << "Angle = " << angleDegree << std::endl;

      meanXError += std::abs( -15.0 - offset[0] );
      meanYError += std::abs( 0.0001 - offset[1] );
      meanZError += std::abs( 0.0001 - offset[2] );
      meanAngleError += std::abs(10.0 - angleDegree);

      myfile.close();

      // condition for destroying smartPointers only
      // when new executions will come afterwards
      if (y < nTimes - 1){
              optimizer.~SmartPointer();
              observer.~SmartPointer();
              registration.~SmartPointer();
              metric.~SmartPointer();
      }
   }

  // Printing out results
  meanXError = meanXError/nTimes;
  meanYError = meanYError/nTimes;
  meanZError = meanZError/nTimes;
  meanAngleError = meanAngleError/nTimes;
  meanNumberOfIterations = meanNumberOfIterations/nTimes;
  meanMetricValue = meanMetricValue/nTimes;
  meanBuildupError = meanXError/(15.0) + meanYError/0.0001 + meanZError/0.0001 + meanAngleError/10.0;

  std::cout << std::endl;
  std::cout << " Result          = " << std::endl;
  std::cout << " qValue          = " << qValue << std::endl;
  std::cout << " Iterations      = " << meanNumberOfIterations<< std::endl;
  std::cout << " meanXError      = " << meanXError<< std::endl;
  std::cout << " meanYError      = " << meanYError<< std::endl;
  std::cout << " meanZError      = " << meanZError<< std::endl;
  std::cout << " meanAngleError      = " << meanAngleError<< std::endl;
  std::cout << " Total Error  = " << meanBuildupError << std::endl;
  std::cout << " Mean Metric Value  = " << meanMetricValue << std::endl;
  std::cout << std::endl;

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


  typedef  unsigned char  OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  typedef itk::CastImageFilter<
                        FixedImageType,
                        OutputImageType > CastFilterType;

  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();

  writer->SetFileName( "OutPut.mha" );

  caster->SetInput( resample->GetOutput() );
  writer->SetInput( caster->GetOutput()   );
  writer->Update();


  // Generate checkerboards before and after registration
  //
  typedef itk::CheckerBoardImageFilter< FixedImageType > CheckerBoardFilterType;

  CheckerBoardFilterType::Pointer checker = CheckerBoardFilterType::New();

  // Inputting the fied image and the resampler filter
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

  writer->SetFileName( "CheckBoardBefore.mha" );
  writer->Update();

  // After registration =================
  // Set the last transformation obtainned in the registrations executions

  resample->SetTransform( registrationPass->GetTransform() );
  writer->SetFileName( "CheckBoardAfter.mha" );
  writer->Update();

  return EXIT_SUCCESS;
}
