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
#ifndef itkNormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader_h
#define itkNormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader_h

#include "itkImageToImageMetricv4GetValueAndDerivativeThreader.h"

#include <mutex>

namespace itk
{

/** \class NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader
 * \brief Processes points for NormalizedMachadoMutualInformationImageToImageMetricv4 \c
 * GetValueAndDerivative.
 *
 * \ingroup ITKMetricsv4
 */
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TNormalizedMachadoMutualInformationMetric>
class ITK_TEMPLATE_EXPORT NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader
  : public ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader);

  /** Standard class type aliases. */
  using Self = NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader;
  using Superclass = ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkTypeMacro(NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader,
               ImageToImageMetricv4GetValueAndDerivativeThreader);

  itkNewMacro(Self);

  using DomainType = typename Superclass::DomainType;
  using AssociateType = typename Superclass::AssociateType;

  using ImageToImageMetricv4Type = typename Superclass::ImageToImageMetricv4Type;
  using VirtualPointType = typename Superclass::VirtualPointType;
  using VirtualIndexType = typename Superclass::VirtualIndexType;
  using FixedImagePointType = typename Superclass::FixedImagePointType;
  using FixedImageIndexType = typename Superclass::FixedImageIndexType;
  using FixedImagePixelType = typename Superclass::FixedImagePixelType;
  using FixedImageGradientType = typename Superclass::FixedImageGradientType;
  using MovingImagePointType = typename Superclass::MovingImagePointType;
  using MovingImagePixelType = typename Superclass::MovingImagePixelType;
  using MovingImageGradientType = typename Superclass::MovingImageGradientType;
  using MeasureType = typename Superclass::MeasureType;
  using DerivativeType = typename Superclass::DerivativeType;
  using DerivativeValueType = typename Superclass::DerivativeValueType;
  using NumberOfParametersType = typename Superclass::NumberOfParametersType;

  using MovingTransformType = typename ImageToImageMetricv4Type::MovingTransformType;

  using PDFValueType = typename TNormalizedMachadoMutualInformationMetric::PDFValueType;
  using JointPDFType = typename TNormalizedMachadoMutualInformationMetric::JointPDFType;
  using JointPDFRegionType = typename TNormalizedMachadoMutualInformationMetric::JointPDFRegionType;
  using JointPDFIndexType = typename TNormalizedMachadoMutualInformationMetric::JointPDFIndexType;
  using JointPDFValueType = typename TNormalizedMachadoMutualInformationMetric::JointPDFValueType;
  using JointPDFSizeType = typename TNormalizedMachadoMutualInformationMetric::JointPDFSizeType;
  using JointPDFDerivativesType = typename TNormalizedMachadoMutualInformationMetric::JointPDFDerivativesType;
  using JointPDFDerivativesIndexType = typename TNormalizedMachadoMutualInformationMetric::JointPDFDerivativesIndexType;
  using JointPDFDerivativesValueType = typename TNormalizedMachadoMutualInformationMetric::JointPDFDerivativesValueType;
  using JointPDFDerivativesRegionType = typename TNormalizedMachadoMutualInformationMetric::JointPDFDerivativesRegionType;
  using JointPDFDerivativesSizeType = typename TNormalizedMachadoMutualInformationMetric::JointPDFDerivativesSizeType;

  using CubicBSplineFunctionType = typename TNormalizedMachadoMutualInformationMetric::CubicBSplineFunctionType;
  using CubicBSplineDerivativeFunctionType =
    typename TNormalizedMachadoMutualInformationMetric::CubicBSplineDerivativeFunctionType;

  using JacobianType = typename TNormalizedMachadoMutualInformationMetric::JacobianType;

protected:
  NormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader()
    : m_MachadoAssociate(nullptr)
  {}

  void
  BeforeThreadedExecution() override;

  void
  AfterThreadedExecution() override;

  /** This function computes the local voxel-wise contribution of
   *  the metric to the global integral of the metric/derivative.
   */
  bool
  ProcessPoint(const VirtualIndexType &        virtualIndex,
               const VirtualPointType &        virtualPoint,
               const FixedImagePointType &     mappedFixedPoint,
               const FixedImagePixelType &     mappedFixedPixelValue,
               const FixedImageGradientType &  mappedFixedImageGradient,
               const MovingImagePointType &    mappedMovingPoint,
               const MovingImagePixelType &    mappedMovingPixelValue,
               const MovingImageGradientType & mappedMovingImageGradient,
               MeasureType &                   metricValueReturn,
               DerivativeType &                localDerivativeReturn,
               const ThreadIdType              threadId) const override;

  /** Compute PDF derivative contribution for each parameter of a displacement field. */
  virtual void
  ComputePDFDerivativesLocalSupportTransform(const JacobianType &            jacobian,
                                             const MovingImageGradientType & movingGradient,
                                             const PDFValueType &            cubicBSplineDerivativeValue,
                                             DerivativeValueType *           localSupportDerivativeResultPtr) const;

private:
  /** Internal pointer to the Machado metric object in use by this threader.
   *  This will avoid costly dynamic casting in tight loops. */
  TNormalizedMachadoMutualInformationMetric * m_MachadoAssociate;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkNormalizedMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx"
#endif

#endif
