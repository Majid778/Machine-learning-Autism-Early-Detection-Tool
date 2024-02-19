# Capstone Project Fall 2023 - NYU Abu Dhabi

## Project Overview
This capstone project, led by Majid Ibrahim, investigates the relationship between the gut microbiome and Autism Spectrum Disorder (ASD) using machine learning. It focuses on the urgent need for early and accurate ASD diagnosis, given the limitations of current diagnostic methods and disparities in care due to costs and accessibility.

## Background
Autism Spectrum Disorder (ASD) is linked to socialization, communication, and behavioral issues. It has been connected to the gut microbiome, which includes bacteria and other organisms in the digestive tract. The project aims to develop cost-effective ASD diagnosis and treatment options based on gut microbiota and machine learning, improving long-term outcomes.

## Project Objectives
1. Establishing machine learning algorithms to predict ASD using gut microbiome data.
2. Identifying biomarkers to understand ASD’s biological underpinnings and suggesting potential pathways for early detection and intervention.
3. Creating a diagnostic tool for early detection of ASD.

## Experimental Results

### Model Performance
- Naive Bayes model achieved outstanding results with the ANOVA F-value feature selection, scoring perfect 1.0000 in both F1 Score and AUC, along with 98.7500 percent accuracy.
- Random Forest model stood out in Mutual Information feature selection with the highest F1 Score and AUC, both at 1.0000, and an accuracy of 97.5000 percent.

### Significant Features
- Two feature selection methods, ANOVA f-test and Mutual Information Gain, were employed to identify crucial features for the machine learning models aimed at detecting ASD.
- 25 features for the ANOVA f-test and 21 for Mutual Information Gain were focused on as these sets contained the most predictive OTUs for ASD.
- Biomarkers identified included Bacteroides, Ruminococcaceae, Lachnospiraceae (specifically Blautia genus), Coprococcus, and Clostridiaceae.
- Ruminococcaceae and Clostridiaceae members were found to be highly abundant in control cases, whereas Blautia was abundant in ASD cases.

## Conclusion
This research advances the use of machine learning in medical science, especially in the context of neurological disorders like ASD. It enhances scientific understanding of ASD and paves the way for practical applications that can profoundly impact individuals affected by this condition.
