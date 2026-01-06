export type UserData = {
  id: string;
  name: string;
  email: string;
};

export type UserMeasurements = {
  height: number; // in cm
  weight: number; // in kg
  chest?: number; // in cm
  waist?: number; // in cm
  hips?: number; // in cm
  inseam?: number; // in cm
  shoulderWidth?: number; // in cm
  sleeveLength?: number; // in cm
};

export type AppFlowState = 
  | "login"
  | "measurements"
  | "video-capture"
  | "recommendations"
  | "dev-legacy";
