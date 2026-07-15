# Hybrid ABSA System

A hybrid Aspect-Based Sentiment Analysis pipeline for extracting
business insights from customer reviews.

The system combines:

- local ABSA model for fast inference;
- confidence-based routing;
- an LLM fallback for uncertain predictions;
- human review and correction;
- feedback logging for future model retraining.

## Problem

Traditional sentiment analysis assigns one sentiment to the entire review.
However, a customer may praise the food while criticising the service.

This project identifies individual aspects and determines sentiment
for each aspect separately.