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
#ifndef itkMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader_h
#define itkMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader_h

#include "itkImageToImageMetricv4GetValueAndDerivativeThreader.h"

#include <mutex>

namespace itk
{

/** \class MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader
 * \brief Processes points for MachadoMutualInformationImageToImageMetricv4 \c
 * GetValueAndDerivative.
 *
 * \ingroup ITKMetricsv4
 */
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TMachadoMutualInformationMetric>
class ITK_TEMPLATE_EXPORT MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader
  : public ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader);

  /** Standard class type aliases. */
  using Self = MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader;
  using Superclass = ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkTypeMacro(MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader,
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

  using PDFValueType = typename TMachadoMutualInformationMetric::PDFValueType;
  using JointPDFType = typename TMachadoMutualInformationMetric::JointPDFType;
  using JointPDFRegionType = typename TMachadoMutualInformationMetric::JointPDFRegionType;
  using JointPDFIndexType = typename TMachadoMutualInformationMetric::JointPDFIndexType;
  using JointPDFValueType = typename TMachadoMutualInformationMetric::JointPDFValueType;
  using JointPDFSizeType = typename TMachadoMutualInformationMetric::JointPDFSizeType;
  using JointPDFDerivativesType = typename TMachadoMutualInformationMetric::JointPDFDerivativesType;
  using JointPDFDerivativesIndexType = typename TMachadoMutualInformationMetric::JointPDFDerivativesIndexType;
  using JointPDFDerivativesValueType = typename TMachadoMutualInformationMetric::JointPDFDerivativesValueType;
  using JointPDFDerivativesRegionType = typename TMachadoMutualInformationMetric::JointPDFDerivativesRegionType;
  using JointPDFDerivativesSizeType = typename TMachadoMutualInformationMetric::JointPDFDerivativesSizeType;

  using CubicBSplineFunctionType = typename TMachadoMutualInformationMetric::CubicBSplineFunctionType;
  using CubicBSplineDerivativeFunctionType =
    typename TMachadoMutualInformationMetric::CubicBSplineDerivativeFunctionType;

  using JacobianType = typename TMachadoMutualInformationMetric::JacobianType;

protected:
  MachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader()
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
  TMachadoMutualInformationMetric * m_MachadoAssociate;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMachadoMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx"
#endif

#endif
