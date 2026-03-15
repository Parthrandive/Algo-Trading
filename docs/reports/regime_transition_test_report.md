# Regime Transition Test Report

Date: March 15, 2026
Owner: Regime Agent (Phase 2 Week 2 Day 5)

## Objective
Validate that regime transitions are detected around major structural breaks while avoiding unstable switching in calm periods.

## Events Tested
- 2008 Financial Crisis (anchor date: September 15, 2008)
- 2013 Taper Tantrum (anchor date: June 1, 2013)
- 2020 COVID Crash (anchor date: March 15, 2020)
- RBI policy-shift proxy windows via macro directional flag transitions

## Method
- Train HMM baseline on a long synthetic replay with embedded break windows.
- Decode daily regime path.
- Measure transition density inside event windows and compare against calm windows.

## Result Summary
- Transition points are present in each event window.
- Average transition density is higher in event windows than calm windows.
- Crisis labeling dominates during the highest-volatility slices.

## Stability Notes
- Outside break windows, regime switching remains comparatively lower.
- OOD warning/alien logic is available to reduce risk when transitions become unstable.

## Decision
Transition validation criteria for Day 5 are satisfied for baseline handoff.
