import React from 'react';
import { UrbanAGBResponse } from '../services/urbanAgbService';

interface SimpleForecastProps {
  analysisResult: UrbanAGBResponse;
}

const SimpleForecast: React.FC<SimpleForecastProps> = ({ analysisResult }) => {
  const { forecasting } = analysisResult;

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="text-2xl">📈</div>
        <h3 className="text-2xl font-bold text-white">3-YEAR FORECAST</h3>
      </div>

      {/* Forecast Years - CORRECT ORDER */}
      <div className="space-y-4 mb-6">
        {/* Year 1 (2026) */}
        <div className="flex justify-between items-center">
          <span className="text-gray-300 text-lg">Year 1 (2026)</span>
          <span className="text-green-400 text-xl font-bold">
            {forecasting.year_1.toFixed(1)} Mg/ha
          </span>
        </div>

        {/* Year 2 (2027) */}
        <div className="flex justify-between items-center">
          <span className="text-gray-300 text-lg">Year 2 (2027)</span>
          <span className="text-green-400 text-xl font-bold">
            {forecasting.year_2.toFixed(1)} Mg/ha
          </span>
        </div>

        {/* Year 3 (2028) */}
        <div className="flex justify-between items-center">
          <span className="text-gray-300 text-lg">Year 3 (2028)</span>
          <span className="text-green-400 text-xl font-bold">
            {forecasting.year_3.toFixed(1)} Mg/ha
          </span>
        </div>
      </div>

      {/* Growth Rate */}
      <div className="pt-4 border-t border-gray-700">
        <div className="flex justify-between items-center">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg">📊</span>
              <span className="text-white font-semibold">Growth Rate</span>
            </div>
            <div className="text-gray-400 text-sm">Annual biomass increase</div>
          </div>
          <span className="text-green-400 text-2xl font-bold">
            {(forecasting.growth_rate * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default SimpleForecast;