// services/processing/lidarProcessing.js
export const processLidarData = async (rawData) => {
    const worker = new Worker('/workers/lidarWorker.js');
    
    return new Promise((resolve, reject) => {
      worker.postMessage({
        type: 'PROCESS_LIDAR',
        data: rawData
      });
  
      worker.onmessage = (e) => {
        if (e.data.type === 'PROCESS_COMPLETE') {
          resolve(e.data.result);
        } else if (e.data.type === 'ERROR') {
          reject(e.data.error);
        }
      };
    });
  };