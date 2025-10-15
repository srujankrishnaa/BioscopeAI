import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        magnetize: "bg-neon-100 text-green hover:bg-neon-80 shadow-xl hover:scale-105 transform transition-all duration-300",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        xl: "h-14 rounded-full px-12 py-5 text-xl font-bold",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface MagnetizeButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  particleCount?: number
}

const MagnetizeButton = React.forwardRef<HTMLButtonElement, MagnetizeButtonProps>(
  ({ className, variant, size, particleCount = 20, children, ...props }, ref) => {
    const buttonRef = React.useRef<HTMLButtonElement>(null)
    const [particles, setParticles] = React.useState<Array<{ id: number; x: number; y: number; vx: number; vy: number; life: number }>>([])

    React.useImperativeHandle(ref, () => buttonRef.current!)

    const createParticles = React.useCallback((e: React.MouseEvent) => {
      if (!buttonRef.current) return
      
      const rect = buttonRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      const newParticles = Array.from({ length: particleCount }, (_, i) => ({
        id: Date.now() + i,
        x,
        y,
        vx: (Math.random() - 0.5) * 8,
        vy: (Math.random() - 0.5) * 8,
        life: 1,
      }))
      
      setParticles(prev => [...prev, ...newParticles])
    }, [particleCount])

    React.useEffect(() => {
      const interval = setInterval(() => {
        setParticles(prev => 
          prev
            .map(particle => ({
              ...particle,
              x: particle.x + particle.vx,
              y: particle.y + particle.vy,
              vx: particle.vx * 0.98,
              vy: particle.vy * 0.98,
              life: particle.life - 0.02,
            }))
            .filter(particle => particle.life > 0)
        )
      }, 16)
      
      return () => clearInterval(interval)
    }, [])

    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={buttonRef}
        onMouseDown={createParticles}
        {...props}
      >
        {children}
        
        {/* Green Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {particles.map(particle => (
            <div
              key={particle.id}
              className="absolute w-1 h-1 bg-neon-100 rounded-full"
              style={{
                left: particle.x,
                top: particle.y,
                opacity: particle.life,
                transform: `scale(${particle.life})`,
                boxShadow: `0 0 ${particle.life * 4}px rgba(85, 221, 74, ${particle.life})`,
              }}
            />
          ))}
        </div>
      </button>
    )
  }
)

MagnetizeButton.displayName = "MagnetizeButton"

export { MagnetizeButton, buttonVariants }