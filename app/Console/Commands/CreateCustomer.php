<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\User;
use Illuminate\Support\Facades\Hash;

class CreateCustomer extends Command
{
    protected $signature = 'create:customer {name} {email} {phone} {username} {password}';
    protected $description = 'Create a new customer user';

    public function handle()
    {
        $name = $this->argument('name');
        $email = $this->argument('email');
        $phone = $this->argument('phone');
        $username = $this->argument('username');
        $password = $this->argument('password');

        // Check if email or username already exists
        if (User::where('email', $email)->exists()) {
            $this->error("Email {$email} already exists!");
            return 1;
        }

        if (User::where('username', $username)->exists()) {
            $this->error("Username {$username} already exists!");
            return 1;
        }

        // Create the customer
        $customer = User::create([
            'name' => $name,
            'email' => $email,
            'phone_number' => $phone,
            'username' => $username,
            'password' => Hash::make($password),
            'role' => 'customer'
        ]);

        $this->info("✅ Customer created successfully!");
        $this->info("ID: {$customer->id}");
        $this->info("Name: {$customer->name}");
        $this->info("Email: {$customer->email}");
        $this->info("Username: {$customer->username}");
        $this->info("Phone: {$customer->phone_number}");
        
        return 0;
    }
}
