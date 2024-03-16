import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, TouchableOpacity, SafeAreaView, Animated } from 'react-native';
import { useRef, useState } from 'react';
import { useEffect } from 'react';
import { Camera, CameraType } from 'expo-camera'
import { Image } from 'expo-image';
import * as FileSystem from 'expo-file-system'
import { Buffer } from "buffer";
import AnimatedLoader from 'react-native-animated-loader';
import {
  useFonts,
  MontserratAlternates_600SemiBold,
} from '@expo-google-fonts/montserrat-alternates';
import AppLoading from 'expo-app-loading';

export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [found, setFound] = useState(false)
  const [flash, setFlash] = useState("off")
  const [loading, setLoading] = useState(false)
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [image, setImage] = useState(null)
  const [guess, setGuess] = useState("")
  const [uri, seturi] = useState("")
  const cameraref = useRef(null)
  let [fontsLoaded] = useFonts({MontserratAlternates_600SemiBold})
  let intervalID

  const getGuess = async () => {
    const base64 = await FileSystem.readAsStringAsync(image, { encoding: 'base64'})
    let my_bytes = Buffer.from(base64, "base64")
    const r = await fetch(image)
    const blob = await r.blob()
    var headers = {
      'Content-Type': 'content_type'
    }
    var reader = new FileReader()
    reader.onload = () => {
      var api = 'https://963c-2603-7081-1601-a4bd-9bc-47d8-8a3f-1b7a.ngrok-free.app/guess'
      const response = async () => {
        return await fetch(api, {
          method: 'POST',
          headers: headers,
          body: reader.result
        }).catch(function(error) {console.log(error)})
      }
      response().then(resp => resp.json()).then(dat => {
        setGuess(dat.answer.slice(10))
        setLoading(false)
        setFound(true)
        setTimeout(() => {
          setFound(false)
        }, 3000)
      }).catch(function(error){console.log(error)})
    }
    reader.readAsDataURL(blob)
  }

  useEffect(() => {
    if(!permission) requestPermission();
    if(image){
      getGuess()
    }
  }, [image])

  const takePicture = async () => {
    if(cameraref){
      const data = await cameraref.current.takePictureAsync();
      setLoading(true)
      setImage(data.uri)
    }
  }

  if(!fontsLoaded){
    return <Text>Still Loading</Text>
  }
  else{
    return (
      <View style={styles.container}>
        {
          permission != null ? 
          <Camera style={styles.camera} type={type} ref={cameraref} onPress={takePicture} flashMode={flash}>
            <SafeAreaView style={styles.title}>
              <Image source={require('./assets/Doggy Vision.svg')} contentFit='contain' style={{marginLeft: 'auto', marginRight: 'auto', height: 200, width: 300}}/>
            </SafeAreaView>
            {
              found === true ? 
              <SafeAreaView style={styles.match}>
                <Text style={{fontSize: 16, color: '#FFFFFF', textAlign: 'center'}}>Match Found!</Text>
              </SafeAreaView>
              :
              loading === true ?
              <SafeAreaView style={{bottom: 0, marginBottom: 300}}>
                <AnimatedLoader source={require('./assets/Animation - 1710556868605.json')} visible={true} speed={1} animationStyle={styles.lottie}/>
              </SafeAreaView>
              :
              <></>
            }
            <SafeAreaView style={{position: 'absolute', bottom: 0, marginBottom: 180, display: 'flex', flexDirection: 'row', justifyContent: 'center'}}>
              <TouchableOpacity onPress={takePicture} style={{width: '70%', height: '100%'}}>
                <Image source={require('./assets/camera - white.svg')} style={{width: 100, height: 100, marginLeft: '55%'}} contentFit='contain'/>
              </TouchableOpacity>
              {
              flash === "on" ?
              <TouchableOpacity onPress={() => {setFlash("off")}} style={{width: '30%', height: '100%', marginTop: 30}}>
                <Image source={require('./assets/flash on - white.svg')} style={{width: 50, height: 50}}/>
              </TouchableOpacity>
              :
              <TouchableOpacity onPress={() => {setFlash("on")}} style={{width: '30%', height: '100%', marginTop: 30}}>
                <Image source={require('./assets/flash off - white.svg')} style={{width: 50, height: 50}}/>
              </TouchableOpacity>
              }
            </SafeAreaView>
            <SafeAreaView style={styles.answer}>
              <Text style={{textAlign: 'center', color: '#87B35B', fontSize: 18}}>Dog Breed Identification</Text>
              <Text style={{textAlign: 'center', color: '#FFFFFF', fontSize: 24}}>{guess}</Text>
            </SafeAreaView>
          </Camera>
          : <Text>This Application requires the camera to be used</Text>
        }
        <StatusBar style="auto" />
      </View>
    );
  }
  
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: 'MontserratAlternates_600SemiBold'
  },
  camera: {
    height: '100%',
    width: '100%',
    alignItems: 'center'
  },
  button: {
    height: '100%',
    width: '50%',
  },
  answer: {
    width: '100%',
    height: '20%',
    backgroundColor: '#32521E',
    position: 'absolute',
    bottom: 0,
    borderTopRightRadius: 25,
    borderTopLeftRadius: 25,
    justifyContent: 'center',
    gap: 50
  },
  title: {
    width: '100%',
    height: '15%',
    backgroundColor: '#32521E',
    borderBottomLeftRadius: 25,
    borderBottomRightRadius: 25,
    justifyContent: 'center'
  },
  match: {
    position: 'absolute',
    bottom: 0,
    marginBottom: 300,
    width: 200,
    height: 50,
    backgroundColor: '#5B8B39',
    borderRadius: 32,
    justifyContent: 'center'
  },
  lottie: {
    width: 100,
    height: 100
  }
});
