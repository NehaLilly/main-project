import React, { useState } from 'react';
import { LogOut, Users, AlertTriangle, Clock, Eye, BarChart3, Shield } from 'lucide-react';

interface AdminDashboardProps {
  onLogout: () => void;
}

interface Student {
  id: string;
  name: string;
  status: 'active' | 'completed' | 'flagged';
  timeRemaining: number;
  alertCount: number;
  currentQuestion: number;
  suspicious: boolean;
}

export default function AdminDashboard({ onLogout }: AdminDashboardProps) {
  const [selectedStudent, setSelectedStudent] = useState<string | null>(null);
  
  // Mock student data
  const students: Student[] = [
    {
      id: '1',
      name: 'John Doe',
      status: 'active',
      timeRemaining: 2400,
      alertCount: 2,
      currentQuestion: 3,
      suspicious: false,
    },
    {
      id: '2',
      name: 'Jane Smith',
      status: 'flagged',
      timeRemaining: 2100,
      alertCount: 7,
      currentQuestion: 2,
      suspicious: true,
    },
    {
      id: '3',
      name: 'Mike Johnson',
      status: 'active',
      timeRemaining: 2700,
      alertCount: 0,
      currentQuestion: 4,
      suspicious: false,
    },
    {
      id: '4',
      name: 'Sarah Wilson',
      status: 'completed',
      timeRemaining: 0,
      alertCount: 1,
      currentQuestion: 5,
      suspicious: false,
    },
  ];

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'completed': return 'bg-blue-100 text-blue-800';
      case 'flagged': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Shield className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
                <p className="text-gray-600">AI-Powered Exam Monitoring System</p>
              </div>
            </div>
            
            <button
              onClick={onLogout}
              className="flex items-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="w-6 h-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Active Students</p>
                <p className="text-2xl font-bold text-gray-900">
                  {students.filter(s => s.status === 'active').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Flagged</p>
                <p className="text-2xl font-bold text-gray-900">
                  {students.filter(s => s.status === 'flagged').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <BarChart3 className="w-6 h-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Completed</p>
                <p className="text-2xl font-bold text-gray-900">
                  {students.filter(s => s.status === 'completed').length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <div className="p-2 bg-orange-100 rounded-lg">
                <Eye className="w-6 h-6 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Alerts</p>
                <p className="text-2xl font-bold text-gray-900">
                  {students.reduce((sum, s) => sum + s.alertCount, 0)}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Student List */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">Students Overview</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Student
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Progress
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Time Left
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Alerts
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Action
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {students.map((student) => (
                      <tr key={student.id} className={student.suspicious ? 'bg-red-50' : ''}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                              <span className="text-sm font-medium text-blue-800">
                                {student.name.split(' ').map(n => n[0]).join('')}
                              </span>
                            </div>
                            <div className="ml-3">
                              <p className="text-sm font-medium text-gray-900">{student.name}</p>
                              <p className="text-sm text-gray-500">ID: {student.id}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(student.status)}`}>
                            {student.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {student.currentQuestion}/5
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                          {student.status === 'completed' ? '00:00:00' : formatTime(student.timeRemaining)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            student.alertCount > 5 ? 'bg-red-100 text-red-800' :
                            student.alertCount > 0 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {student.alertCount}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => setSelectedStudent(student.id)}
                            className="text-blue-600 hover:text-blue-900"
                          >
                            Monitor
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Monitoring Panel */}
          <div className="bg-white rounded-lg shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Live Monitoring</h2>
            </div>
            <div className="p-6">
              {selectedStudent ? (
                <div>
                  <div className="mb-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-2">
                      Monitoring: {students.find(s => s.id === selectedStudent)?.name}
                    </h3>
                    <div className="bg-gray-800 rounded aspect-video flex items-center justify-center mb-4">
                      <span className="text-gray-400 text-sm">Live Camera Feed</span>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">AI Analysis</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Face Detection</span>
                          <span className="text-green-600">✓ Active</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Gaze Tracking</span>
                          <span className="text-green-600">✓ Normal</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Audio Analysis</span>
                          <span className="text-green-600">✓ Clear</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Recent Activities</h4>
                      <div className="space-y-2 text-sm text-gray-600">
                        <div>• Code submitted at 14:32</div>
                        <div>• Looking at question at 14:30</div>
                        <div>• Started typing at 14:28</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Eye className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">Select a student to monitor</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}