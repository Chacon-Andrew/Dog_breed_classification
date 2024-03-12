import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, TouchableOpacity } from 'react-native';
import { useRef, useState } from 'react';
import { useEffect } from 'react';
import { Camera, CameraType } from 'expo-camera'
import { Image } from 'expo-image';

export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [image, setImage] = useState(null)
  const [guess, setGuess] = useState("")
  const cameraref = useRef(null)

  useEffect(() => {
    if(!permission) requestPermission();
    setGuess("Berri!!!")
    console.log("hit")
  }, [image])

  const takePicture = async () => {
    if(cameraref){
      const data = await cameraref.current.takePictureAsync();
      setImage(data)
    }
  }

  return (
    <View style={styles.container}>
      {
        permission != null ? 
        <Camera style={styles.camera} type={type} ref={cameraref} onPress={takePicture}>
          <View style={{position: 'absolute', bottom: 0, marginBottom: 20}}>
            <TouchableOpacity onPress={takePicture} style={styles.button}>
              <Image source={require('./assets/icons8-camera.svg')} style={{width: 100, height: 100}} contentFit='contain'/>
            </TouchableOpacity>
          </View>
        </Camera>
        : <Text>This Application requires the camera to be used</Text>
      }
      <Text>{guess}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    height: '80%',
    width: '100%',
    alignItems: 'center'
  },
  button: {
    height: '100%',
    width: '100%',
  }
});
