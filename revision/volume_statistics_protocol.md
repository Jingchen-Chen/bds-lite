# Volume statistics eligibility clarification

Registered before inspecting new volume effects. Read with volume_evaluation_protocol.md.

Surface eligibility clarification before inspecting the new volume effects:
a primary patient/case surface endpoint is eligible only when every required
class/frame measurement in both methods and all three seeds is finite. Report
the excluded-unit count and retain an all-unit diagonal-penalty sensitivity.
Do not improve an endpoint by silently dropping a method's missed class from
its within-case average. No surface SESOI judgement is assigned on uncalibrated
3D index grids. This clarification changes no stored volume computation.
