import { useState } from "react"
import { useAuthStore, selectCurrentUser } from "@/stores/auth-store"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"

export default function AuthGate({ children }) {
  const login = useAuthStore((state) => state.login)
  const register = useAuthStore((state) => state.register)
  const user = useAuthStore(selectCurrentUser)
  const [mode, setMode] = useState("login")
  const [formState, setFormState] = useState({ username: "", password: "" })
  const [status, setStatus] = useState({ state: "idle", message: "" })

  const handleChange = (event) => {
    const { name, value } = event.target
    setFormState((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setStatus({ state: "pending", message: "" })
    const action = mode === "login" ? login : register
    const result = await action({ ...formState })
    if (!result.success) {
      setStatus({ state: "error", message: result.error })
      return
    }
    setFormState({ username: "", password: "" })
    setStatus({ state: "idle", message: "" })
  }

  if (user) {
    return children
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-background via-background to-muted/40 p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>{mode === "login" ? "Log in" : "Create account"}</CardTitle>
          <CardDescription>
            {mode === "login"
              ? "Use any username/password pair. Data is stored locally in your browser."
              : "Register a lightweight local account. No external network calls are made."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                name="username"
                autoComplete="username"
                value={formState.username}
                onChange={handleChange}
                placeholder="e.g. admin"
              />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                value={formState.password}
                onChange={handleChange}
                placeholder="Enter a password"
              />
            </div>
            {status.state === "error" && <p className="text-sm text-destructive">{status.message}</p>}
            <Button type="submit" className="w-full" disabled={status.state === "pending"}>
              {status.state === "pending" ? "Please wait" : mode === "login" ? "Login" : "Register"}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col gap-3">
          <Button variant="ghost" type="button" className="w-full text-sm" onClick={() => setMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "Need an account? Register" : "Have an account? Login"}
          </Button>
          <p className="text-center text-xs text-muted-foreground">
            Accounts, conversations, endpoints, and tools all live in localStorage until a backend is connected.
          </p>
        </CardFooter>
      </Card>
    </div>
  )
}
