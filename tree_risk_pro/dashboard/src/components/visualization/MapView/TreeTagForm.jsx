// src/components/visualization/MapView/TreeTagForm.jsx

import React, { useState } from 'react';

/**
 * A form component for entering tree details when tagging trees on the map
 */
const TreeTagForm = ({ onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    species: '',
    height: '',
    canopyWidth: '',
    diameter: '',
    notes: ''
  });
  
  const speciesOptions = [
    'Live Oak',
    'Post Oak',
    'Red Oak',
    'Shumard Oak',
    'White Oak',
    'Cedar Elm',
    'Southern Magnolia',
    'Bald Cypress',
    'Chinese Pistache',
    'Pecan',
    'Other'
  ];
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };
  
  return (
    <div className="bg-white rounded-md shadow-md p-3 w-full max-w-md">
      <h3 className="text-sm font-medium mb-2">New Tree Details</h3>
      
      <form onSubmit={handleSubmit} className="space-y-2">
        {/* Species */}
        <div>
          <label className="block text-xs text-gray-600 mb-1">
            Species <span className="text-red-500">*</span>
          </label>
          <select
            name="species"
            value={formData.species}
            onChange={handleChange}
            required
            className="w-full p-1 text-xs border rounded text-gray-600 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
          >
            <option value="">Select species...</option>
            {speciesOptions.map(species => (
              <option key={species} value={species}>{species}</option>
            ))}
          </select>
        </div>
        
        {/* Tree Measurements - Height and Canopy Width */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-600 mb-1">
              Height (ft)
            </label>
            <input
              type="number"
              name="height"
              value={formData.height}
              onChange={handleChange}
              placeholder="e.g. 25"
              className="w-full p-1 text-xs border rounded text-gray-600 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
          
          <div>
            <label className="block text-xs text-gray-600 mb-1">
              Canopy Width (ft)
            </label>
            <input
              type="number"
              name="canopyWidth"
              value={formData.canopyWidth}
              onChange={handleChange}
              placeholder="e.g. 15"
              className="w-full p-1 text-xs border rounded text-gray-600 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
        </div>
        
        {/* Tree Diameter */}
        <div>
          <label className="block text-xs text-gray-600 mb-1">
            Diameter (inches)
          </label>
          <input
            type="number"
            name="diameter"
            value={formData.diameter}
            onChange={handleChange}
            placeholder="e.g. 18"
            className="w-full p-1 text-xs border rounded text-gray-600 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
          />
        </div>
        
        {/* Notes */}
        <div>
          <label className="block text-xs text-gray-600 mb-1">
            Notes
          </label>
          <textarea
            name="notes"
            value={formData.notes}
            onChange={handleChange}
            rows={2}
            placeholder="Any additional observations..."
            className="w-full p-1 text-xs border rounded text-gray-600 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
          />
        </div>
        
        {/* Form Actions */}
        <div className="flex justify-end space-x-2 mt-3">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
          >
            Cancel
          </button>
          
          <button
            type="submit"
            className="px-3 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600"
          >
            Save Tree
          </button>
        </div>
      </form>
    </div>
  );
};

export default TreeTagForm;