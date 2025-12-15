import React from 'react';
import { FileText, Clock } from 'lucide-react';

export default function QuestionPanel() {
  return (
    <div className="p-6 border-b border-gray-200">
      <div className="flex items-center space-x-2 mb-4">
        <FileText className="w-5 h-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-gray-900">Question 1</h2>
        <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">Algorithm</span>
      </div>

      <div className="prose prose-sm max-w-none">
        <h3 className="text-base font-medium text-gray-900 mb-3">
          Implement Binary Search Algorithm
        </h3>
        
        <p className="text-gray-700 mb-4">
          Write a function that implements the binary search algorithm to find the position of a target value in a sorted array.
        </p>

        <div className="bg-gray-50 p-4 rounded-lg mb-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Requirements:</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• Function should be named <code className="bg-gray-200 px-1 rounded">binarySearch</code></li>
            <li>• Take parameters: <code className="bg-gray-200 px-1 rounded">arr</code> (sorted array) and <code className="bg-gray-200 px-1 rounded">target</code> (value to find)</li>
            <li>• Return the index of the target if found, -1 otherwise</li>
            <li>• Time complexity should be O(log n)</li>
          </ul>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Example:</h4>
          <pre className="text-sm text-blue-800 font-mono">
{`Input: arr = [1, 3, 5, 7, 9, 11], target = 7
Output: 3

Input: arr = [1, 3, 5, 7, 9, 11], target = 4
Output: -1`}
          </pre>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between text-sm text-gray-500">
        <span className="flex items-center space-x-1">
          <Clock className="w-4 h-4" />
          <span>Suggested time: 20 minutes</span>
        </span>
        <span>Points: 25</span>
      </div>
    </div>
  );
}