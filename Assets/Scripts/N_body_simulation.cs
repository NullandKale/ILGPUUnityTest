using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

public class N_body_simulation : MonoBehaviour
{
    private Context context;
    private Accelerator device;
    private HostParticleSystemStructOfArrays ILGPUParticleSystem;

    public float maxWidth;
    public float maxHeight;

    public int body_count;

    public GameObject center_object;
    public GameObject body_prefab;

    private GameObject[] bodies;

    // Start is called before the first frame update
    void Start()
    {
        context = Context.Create(builder => builder.Default().EnableAlgorithms());
        device = context.GetPreferredDevice(preferCPU: false)
                                  .CreateAccelerator(context);
        Debug.Log("Using Device: " + device.Name);

        ILGPUParticleSystem = new HostParticleSystemStructOfArrays(device, body_count, (int)maxWidth, (int)maxHeight);

        bodies = new GameObject[body_count];
        for(int i = 0; i < body_count; i++)
        {
            //Vec3 pos = ILGPUParticleSystem.hostPositions[i];
            //bodies[i] = Instantiate(body_prefab, new Vector3(pos.x, pos.y, pos.z), new Quaternion() ,center_object.transform);
            bodies[i] = Instantiate(body_prefab, new Vector3(), new Quaternion(),center_object.transform);
        }
    }

    // Update is called once per frame
    void Update()
    {
        //ILGPUParticleSystem.particleProcessingKernel(body_count, ILGPUParticleSystem.deviceParticleSystem);
        //ILGPUParticleSystem.updateHostPostitionData();
        //for (int i = 0; i < body_count; i++)
        //{
        //    Vec3 pos = ILGPUParticleSystem.hostPositions[i];
        //    bodies[i].transform.position = new Vector3(pos.x, pos.y, pos.z);
        //}
    }

    public class HostParticleSystemStructOfArrays : System.IDisposable
    {
        public int particleCount;
        public Vec3[] hostPositions;
        public MemoryBuffer1D<Vec3, Stride1D.Dense> positions;
        public MemoryBuffer1D<Vec3, Stride1D.Dense> velocities;
        public MemoryBuffer1D<Vec3, Stride1D.Dense> accelerations;
        public ParticleSystemStructOfArrays deviceParticleSystem;
        public System.Action<Index1D, ParticleSystemStructOfArrays> particleProcessingKernel;

        public HostParticleSystemStructOfArrays(Accelerator device, int particleCount, int width, int height)
        {
            this.particleCount = particleCount;
            hostPositions = new Vec3[particleCount];

            for (int i = 0; i < particleCount; i++)
            {
                hostPositions[i] = new Vec3(Random.value * width, Random.value * height, 1);
            }

            Debug.Log("Generated Positions");

            positions = device.Allocate1D(hostPositions);
            velocities = device.Allocate1D<Vec3>(particleCount);
            accelerations = device.Allocate1D<Vec3>(particleCount);

            Debug.Log("Allocated Cuda Memory");

            velocities.MemSetToZero();
            accelerations.MemSetToZero();

            Debug.Log("Cleared Cuda Memory");

            deviceParticleSystem = new ParticleSystemStructOfArrays(positions, velocities, accelerations, width, height);
            Debug.Log("Created device particle system");

            particleProcessingKernel = device.LoadAutoGroupedStreamKernel<Index1D, ParticleSystemStructOfArrays>(ParticleSystemStructOfArrays.particleKernel);

            Debug.Log("Compiled Kernel");
        }

        public void updateHostPostitionData()
        {
            positions.CopyToCPU(hostPositions);
        }

        public void Dispose()
        {
            positions.Dispose();
            velocities.Dispose();
            accelerations.Dispose();
        }
    }

    public struct ParticleSystemStructOfArrays
    {
        public ArrayView1D<Vec3, Stride1D.Dense> positions;
        public ArrayView1D<Vec3, Stride1D.Dense> velocities;
        public ArrayView1D<Vec3, Stride1D.Dense> accelerations;
        public float gc;
        public Vec3 centerPos;
        public float centerMass;

        public ParticleSystemStructOfArrays(ArrayView1D<Vec3, Stride1D.Dense> positions, ArrayView1D<Vec3, Stride1D.Dense> velocities, ArrayView1D<Vec3, Stride1D.Dense> accelerations, int width, int height)
        {
            this.positions = positions;
            this.velocities = velocities;
            this.accelerations = accelerations;
            gc = 0.001f;
            centerPos = new Vec3(0.5f * width, 0.5f * height, 0);
            centerMass = (float)positions.Length;
        }

        public static void particleKernel(Index1D index, ParticleSystemStructOfArrays p)
        {
            p.updateAcceleration(index);
            p.updatePosition(index);
            p.updateVelocity(index);
        }

        private void updateAcceleration(int ID)
        {
            accelerations[ID] = new Vec3();

            for (int i = 0; i < positions.Length; i++)
            {
                Vec3 otherPos;
                float mass;

                if (i == ID)
                {
                    //creates a mass at the center of the screen
                    otherPos = centerPos;
                    mass = centerMass;
                }
                else
                {
                    otherPos = positions[i];
                    mass = 1f;
                }

                float deltaPosLength = (positions[ID] - otherPos).length();
                float temp = (gc * mass) / XMath.Pow(deltaPosLength, 3f);
                accelerations[ID] += (otherPos - positions[ID]) * temp;
            }
        }

        private void updatePosition(int ID)
        {
            positions[ID] = positions[ID] + velocities[ID] + accelerations[ID] * 0.5f;
        }

        private void updateVelocity(int ID)
        {
            velocities[ID] = velocities[ID] + accelerations[ID];
        }

    }

    public struct Vec3
    {
        public float x;
        public float y;
        public float z;

        public Vec3(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static Vec3 operator +(Vec3 v1, Vec3 v2)
        {
            return new Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        }

        public static Vec3 operator -(Vec3 v1, Vec3 v2)
        {
            return new Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        }

        public static Vec3 operator *(Vec3 v1, float v)
        {
            return new Vec3(v1.x * v, v1.y * v, v1.z * v);
        }

        public float length()
        {
            return XMath.Sqrt(x * x + y * y + z * z);
        }
    }
}
