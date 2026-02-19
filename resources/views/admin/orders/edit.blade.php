@extends('layouts.adminSB')
@section('title', 'Quickbite - Edit Order')

@section('content')
<div class="container">
    <h2>Edit Order</h2>

    @if($errors->any())
        <div class="alert alert-danger">
            <ul>@foreach($errors->all() as $error) <li>{{ $error }}</li> @endforeach</ul>
        </div>
    @endif

    <form action="{{ route('admin.order.update', $order->id) }}" method="POST">
        @csrf
        @method('PUT')

        <div class="mb-3">
            <label>Transaksi</label>
            <select name="transactions_id" class="form-control" required>
                @foreach($transactions as $transaction)
                    <option value="{{ $transaction->id }}" {{ $order->transactions_id == $transaction->id ? 'selected' : '' }}>{{ $transaction->id }}</option>
                @endforeach
            </select>
        </div>

        <div class="mb-3">
            <label>Menu</label>
            <select name="menus_id" class="form-control" required>
                @foreach($menus as $menu)
                    <option value="{{ $menu->id }}" {{ $order->menus_id == $menu->id ? 'selected' : '' }}>{{ $menu->name }}</option>
                @endforeach
            </select>
        </div>

        <div class="mb-3">
            <label>Porsi</label>
            <input type="text" name="portion_size" class="form-control" value="{{ $order->portion_size }}" required>
        </div>

        <div class="mb-3">
            <label>Jumlah</label>
            <input type="number" name="quantity" class="form-control" value="{{ $order->quantity }}" required>
        </div>

        <div class="mb-3">
            <label>Total Harga</label>
            <input type="number" name="total" class="form-control" value="{{ $order->total }}" required>
        </div>

        <div class="mb-3">
            <label>Catatan</label>
            <textarea name="notes" class="form-control">{{ $order->notes }}</textarea>
        </div>

        <button type="submit" class="btn btn-primary">Update</button>
        <a href="{{ route('admin.order.index') }}" class="btn btn-secondary">Kembali</a>
    </form>
</div>
@endsection
